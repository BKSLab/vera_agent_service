from types import SimpleNamespace

import pytest

from app.messaging.consumer import AgentRequestConsumer


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

    def astream_events(self, state, config, version='v2'):
        events = self._events_per_call[self.call_count]
        self.call_count += 1

        async def _generator():
            for item in events:
                if isinstance(item, Exception):
                    raise item
                yield item

        return _generator()


def _token_event(content: str) -> dict:
    return {'event': 'on_chat_model_stream', 'data': {'chunk': SimpleNamespace(content=content)}}


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


async def test_successful_message_streams_tokens_and_acks():
    graph = _FakeGraph([[_token_event('Квота'), _token_event(' 2%.')]])
    sink = _TokenSinkRecorder()
    consumer = _build_consumer(graph, sink)
    message = _FakeMessage(body=b'{"session_id": "s1", "message": "?"}')

    await consumer._handle_message(message)

    assert message.acked
    assert message.nacked_requeue is None
    assert sink.calls == [
        ('s1', {'type': 'token', 'content': 'Квота'}),
        ('s1', {'type': 'token', 'content': ' 2%.'}),
        ('s1', {'type': 'done'}),
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
    message = _FakeMessage(body=b'{"session_id": "s1", "message": "?"}')

    await consumer._handle_message(message)

    assert graph.call_count == 2
    assert message.acked
    assert ('s1', {'type': 'done'}) in sink.calls
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
    message = _FakeMessage(body=b'{"session_id": "s1", "message": "?"}')

    await consumer._handle_message(message)

    assert graph.call_count == 3
    assert message.nacked_requeue is False
    assert not message.acked
    assert sink.calls[-1] == ('s1', {'type': 'error', 'detail': 'Сервис временно недоступен, попробуйте позже.'})


async def test_failure_after_streaming_started_does_not_retry_and_acks():
    """Ключевой инвариант раздела 0.1: сбой после того как хотя бы один
    токен уже ушёл в SSE — не повод для повтора всего сообщения."""
    graph = _FakeGraph([[_token_event('Начало ответа'), RuntimeError('обрыв соединения с LLM')]])
    sink = _TokenSinkRecorder()
    consumer = _build_consumer(graph, sink, retries=3)
    message = _FakeMessage(body=b'{"session_id": "s1", "message": "?"}')

    await consumer._handle_message(message)

    assert graph.call_count == 1
    assert message.acked
    assert message.nacked_requeue is None
    assert sink.calls == [
        ('s1', {'type': 'token', 'content': 'Начало ответа'}),
        ('s1', {'type': 'error', 'detail': 'Произошла ошибка при формировании ответа.'}),
    ]
