from types import SimpleNamespace

import pytest

from app.messaging.consumer import AgentRequestConsumer
from app.observability.request_trace import get_request_trace


class _FakeMessage:
    def __init__(self, body: bytes):
        self.body = body
        self.acked = False
        self.nacked_requeue: bool | None = None

    async def ack(self):
        self.acked = True

    async def nack(self, requeue: bool = True):
        self.nacked_requeue = requeue


class _FakeGraph:
    """`events_per_call[i]` — поток событий, который вернёт i-й по счёту
    вызов `astream_events` (для сценариев ретраев). Элемент-исключение в
    списке — событие не отдаётся, вместо этого поток падает."""

    def __init__(self, events_per_call: list[list]):
        self._events_per_call = events_per_call
        self.call_count = 0
        self.states: list[dict] = []
        self.configs: list[dict] = []

    def astream_events(self, state, config, version='v2'):
        self.states.append(state)
        self.configs.append(config)
        events = self._events_per_call[self.call_count]
        self.call_count += 1

        async def _generator():
            for item in events:
                if isinstance(item, Exception):
                    raise item
                yield item

        return _generator()


class _MutatingFailureGraph:
    def __init__(self):
        self.call_count = 0

    def astream_events(self, state, config, version='v2'):
        self.call_count += 1

        async def _generator():
            trace_data = get_request_trace()
            trace_data.mutating_tool_called = True
            raise RuntimeError('LLM failed after email tool call')
            yield

        return _generator()


def _token_event(content: str, node: str = 'generate_direct') -> dict:
    return {
        'event': 'on_chat_model_stream',
        'metadata': {'langgraph_node': node},
        'data': {'chunk': SimpleNamespace(content=content)},
    }


class _TokenSinkRecorder:
    def __init__(self):
        self.calls: list[tuple[str, dict]] = []

    async def __call__(self, session_id: str, event: dict) -> None:
        self.calls.append((session_id, event))


def _build_consumer(graph: _FakeGraph, sink: _TokenSinkRecorder, retries: int = 3) -> AgentRequestConsumer:
    return AgentRequestConsumer(
        connection_url='amqp://unused',
        queue_name='agent.requests',
        dlq_name='agent.requests.dlq',
        graph=graph,
        token_sink=sink,
        retries=retries,
    )


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    async def _instant_sleep(_seconds):
        return None

    monkeypatch.setattr('app.messaging.consumer.asyncio.sleep', _instant_sleep)


async def test_invalid_payload_goes_to_dlq_without_calling_graph():
    graph = _FakeGraph([])
    sink = _TokenSinkRecorder()
    consumer = _build_consumer(graph, sink)
    message = _FakeMessage(body=b'not valid json')

    await consumer._handle_message(message)

    assert message.nacked_requeue is False
    assert not message.acked
    assert graph.call_count == 0


async def test_payload_without_request_id_goes_to_dlq_without_calling_graph():
    graph = _FakeGraph([])
    sink = _TokenSinkRecorder()
    consumer = _build_consumer(graph, sink)
    message = _FakeMessage(body=b'{"session_id": "s1", "message": "?"}')

    await consumer._handle_message(message)

    assert message.nacked_requeue is False
    assert not message.acked
    assert graph.call_count == 0
    assert sink.calls == []


async def test_successful_message_streams_tokens_and_acks():
    graph = _FakeGraph([[_token_event('Квота'), _token_event(' 2%.')]])
    sink = _TokenSinkRecorder()
    consumer = _build_consumer(graph, sink)
    message = _FakeMessage(body=b'{"session_id": "s1", "request_id": "r1", "message": "?"}')

    await consumer._handle_message(message)

    assert message.acked
    assert message.nacked_requeue is None
    assert sink.calls == [
        ('r1', {'type': 'token', 'content': 'Квота'}),
        ('r1', {'type': 'token', 'content': ' 2%.'}),
        ('r1', {'type': 'done'}),
    ]


async def test_request_id_routes_delivery_without_changing_session_history_key():
    graph = _FakeGraph([[_token_event('Ответ')]])
    sink = _TokenSinkRecorder()
    consumer = _build_consumer(graph, sink)
    message = _FakeMessage(
        body=b'{"session_id": "conversation-1", "request_id": "request-1", "message": "?"}'
    )

    await consumer._handle_message(message)

    assert message.acked
    assert sink.calls == [
        ('request-1', {'type': 'token', 'content': 'Ответ'}),
        ('request-1', {'type': 'done'}),
    ]
    assert graph.states[0]['session_id'] == 'conversation-1'
    assert graph.configs[0] == {'configurable': {'thread_id': 'conversation-1'}}


async def test_streams_only_final_node_tokens_and_ignores_internal_llm_output():
    graph = _FakeGraph(
        [
            [
                _token_event('Внутренний ответ анализа.', node='analyze_intent'),
                _token_event('Финальный ответ.', node='generate_with_context'),
            ]
        ]
    )
    sink = _TokenSinkRecorder()
    consumer = _build_consumer(graph, sink)
    message = _FakeMessage(body=b'{"session_id": "s1", "request_id": "r1", "message": "?"}')

    await consumer._handle_message(message)

    assert message.acked
    assert sink.calls == [
        ('r1', {'type': 'token', 'content': 'Финальный ответ.'}),
        ('r1', {'type': 'done'}),
    ]


async def test_ignores_stream_events_without_confirmed_graph_node():
    graph = _FakeGraph(
        [
            [
                {
                    'event': 'on_chat_model_stream',
                    'data': {'chunk': SimpleNamespace(content='Неизвестный источник')},
                },
                _token_event('Финальный ответ.'),
            ]
        ]
    )
    sink = _TokenSinkRecorder()
    consumer = _build_consumer(graph, sink)
    message = _FakeMessage(body=b'{"session_id": "s1", "request_id": "r1", "message": "?"}')

    await consumer._handle_message(message)

    assert message.acked
    assert sink.calls == [
        ('r1', {'type': 'token', 'content': 'Финальный ответ.'}),
        ('r1', {'type': 'done'}),
    ]


async def test_failure_before_streaming_retries_then_succeeds():
    graph = _FakeGraph(
        [
            [RuntimeError('Redis временно недоступен')],
            [_token_event('Ok')],
        ]
    )
    sink = _TokenSinkRecorder()
    consumer = _build_consumer(graph, sink, retries=3)
    message = _FakeMessage(body=b'{"session_id": "s1", "request_id": "r1", "message": "?"}')

    await consumer._handle_message(message)

    assert graph.call_count == 2
    assert message.acked
    assert ('r1', {'type': 'done'}) in sink.calls
    assert not any(event.get('type') == 'error' for _, event in sink.calls)


async def test_failure_before_streaming_exhausts_retries_goes_to_dlq():
    graph = _FakeGraph(
        [
            [RuntimeError('a')],
            [RuntimeError('b')],
            [RuntimeError('c')],
        ]
    )
    sink = _TokenSinkRecorder()
    consumer = _build_consumer(graph, sink, retries=3)
    message = _FakeMessage(body=b'{"session_id": "s1", "request_id": "r1", "message": "?"}')

    await consumer._handle_message(message)

    assert graph.call_count == 3
    assert message.nacked_requeue is False
    assert not message.acked
    assert sink.calls[-1] == ('r1', {'type': 'error', 'detail': 'Сервис временно недоступен, попробуйте позже.'})


async def test_failure_after_streaming_started_does_not_retry_and_acks():
    """Ключевой инвариант раздела 0.1: сбой после того как хотя бы один
    токен уже ушёл в SSE — не повод для повтора всего сообщения."""
    graph = _FakeGraph([[_token_event('Начало ответа'), RuntimeError('обрыв соединения с LLM')]])
    sink = _TokenSinkRecorder()
    consumer = _build_consumer(graph, sink, retries=3)
    message = _FakeMessage(body=b'{"session_id": "s1", "request_id": "r1", "message": "?"}')

    await consumer._handle_message(message)

    assert graph.call_count == 1
    assert message.acked
    assert message.nacked_requeue is None
    assert sink.calls == [
        ('r1', {'type': 'token', 'content': 'Начало ответа'}),
        ('r1', {'type': 'error', 'detail': 'Произошла ошибка при формировании ответа.'}),
    ]


async def test_failure_after_mutating_tool_call_never_retries_and_acks():
    """Даже без отправленного SSE-токена весь граф нельзя повторять после
    вызова email-тулы: письмо могло быть принято до последующего сбоя."""
    graph = _MutatingFailureGraph()
    sink = _TokenSinkRecorder()
    consumer = _build_consumer(graph, sink, retries=3)
    message = _FakeMessage(
        body=b'{"session_id": "s1", "request_id": "r1", "message": "send"}'
    )

    await consumer._handle_message(message)

    assert graph.call_count == 1
    assert message.acked
    assert message.nacked_requeue is None
    assert sink.calls == [
        (
            'r1',
            {
                'type': 'error',
                'detail': (
                    'Не удалось подтвердить результат отправки консультации. '
                    'Проверьте почту перед новой попыткой.'
                ),
            },
        )
    ]
