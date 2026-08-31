import json
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph

from app.db.models.chat_turn import STATUS_PROCESSING
from app.exceptions.llm import EmptyLlmStreamError
from app.graph.state import AgentState
from app.messaging.consumer import AgentRequestConsumer, TurnPersistenceData
from app.messaging.schemas import AgentRequestMessage
from app.observability.request_trace import get_request_trace
from app.privacy.pii import UnresolvedEmailError, resolve_email_for_tool
from app.services.chat_persistence import (
    START_CLAIMED,
    ChatPersistenceService,
    TurnStartResult,
)


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


class _SessionPiiGraph:
    """Минимально имитирует merge и checkpoint между репликами сессии."""

    def __init__(
        self,
        checkpoint_values: dict | None = None,
        resolve_on_call: dict[int, list[str]] | None = None,
    ):
        self.checkpoint_values = checkpoint_values or {}
        self.resolve_on_call = resolve_on_call or {}
        self.states: list[dict] = []
        self.resolved: list[tuple[str, str | None]] = []

    async def aget_state(self, config):
        return SimpleNamespace(values=self.checkpoint_values)

    def astream_events(self, state, config, version='v2'):
        self.states.append(state)
        call_number = len(self.states)
        previous_messages = list(self.checkpoint_values.get('messages', []))
        self.checkpoint_values = {
            **self.checkpoint_values,
            **state,
            'messages': [*previous_messages, *state['messages']],
        }

        async def _generator():
            for alias in self.resolve_on_call.get(call_number, []):
                try:
                    resolved = resolve_email_for_tool(alias)
                except UnresolvedEmailError:
                    resolved = None
                self.resolved.append((alias, resolved))
            yield _token_event(f'Ответ {call_number}')

        return _generator()


def _raw_model_event(content: str, node: str = 'generate_direct') -> dict:
    return {
        'event': 'on_chat_model_stream',
        'metadata': {'langgraph_node': node},
        'data': {'chunk': SimpleNamespace(content=content)},
    }


def _token_event(content: object, node: str = 'generate_direct') -> dict:
    """Завершённый snapshot с проверенным AIMessage для одного SSE token."""
    return {
        'event': 'on_chain_end',
        'parent_ids': ['graph-run'],
        'metadata': {'langgraph_node': node},
        'data': {'output': {'messages': [AIMessage(content=content)]}},
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
    graph = _FakeGraph([[_token_event('Квота 2%.')]])
    sink = _TokenSinkRecorder()
    consumer = _build_consumer(graph, sink)
    message = _FakeMessage(
        body=b'{"session_id": "s1", "request_id": "r1", "user_id": "u1", "message": "?"}'
    )

    await consumer._handle_message(message)

    assert message.acked
    assert message.nacked_requeue is None
    assert sink.calls == [
        ('r1', {'type': 'token', 'content': 'Квота 2%.'}),
        ('r1', {'type': 'done', 'used_knowledge_base': False}),
    ]


async def test_request_id_routes_delivery_without_changing_session_history_key():
    graph = _FakeGraph([[_token_event('Ответ')]])
    sink = _TokenSinkRecorder()
    consumer = _build_consumer(graph, sink)
    message = _FakeMessage(
        body=b'{"session_id": "conversation-1", "request_id": "request-1", "user_id": "u1", "message": "?"}'
    )

    await consumer._handle_message(message)

    assert message.acked
    assert sink.calls == [
        ('request-1', {'type': 'token', 'content': 'Ответ'}),
        ('request-1', {'type': 'done', 'used_knowledge_base': False}),
    ]
    assert graph.states[0]['session_id'] == 'conversation-1'
    assert graph.states[0]['messages'][0].id == 'request-1'
    assert graph.configs[0] == {'configurable': {'thread_id': 'conversation-1'}}


async def test_consumer_sends_only_redacted_message_to_graph():
    graph = _FakeGraph([[_token_event('Ответ')]])
    sink = _TokenSinkRecorder()
    consumer = _build_consumer(graph, sink)
    full_name = 'Иванов Иван Иванович'
    email = 'ivan.petrov@example.com'
    message = _FakeMessage(
        body=json.dumps(
            {
                'session_id': 'conversation-1',
                'request_id': 'request-1',
                'user_id': 'u1',
                'message': f'Я {full_name}, ответьте на {email}',
            },
            ensure_ascii=False,
        ).encode(),
    )

    await consumer._handle_message(message)

    graph_message = graph.states[0]['messages'][0].content
    assert full_name not in graph_message
    assert email not in graph_message
    assert graph_message == 'Я [ФИО_1], ответьте на [EMAIL_1]'
    assert graph.states[0]['pii_aliases'] == {
        '[ФИО_1]': ['PERSON', full_name],
        '[EMAIL_1]': ['EMAIL', email],
    }


async def test_email_alias_is_hydrated_and_resolved_in_next_turn():
    email = 'first@example.com'
    graph = _SessionPiiGraph(resolve_on_call={2: ['[EMAIL_1]']})
    consumer = _build_consumer(graph, _TokenSinkRecorder())

    first_payload = AgentRequestMessage(
        session_id='session-1',
        request_id='request-1',
        user_id='user-1',
        message=f'Отправьте консультацию на {email}',
    )
    second_payload = AgentRequestMessage(
        session_id='session-1',
        request_id='request-2',
        user_id='user-1',
        message='Да, отправьте',
    )

    first_answer = ''.join(
        [
            chunk
            async for chunk in consumer._stream_answer(
                first_payload,
                TurnPersistenceData(),
            )
        ]
    )
    second_answer = ''.join(
        [
            chunk
            async for chunk in consumer._stream_answer(
                second_payload,
                TurnPersistenceData(),
            )
        ]
    )

    assert first_answer == 'Ответ 1'
    assert second_answer == 'Ответ 2'
    assert graph.states[0]['messages'][0].content.endswith('[EMAIL_1]')
    assert graph.states[1]['messages'][0].content == 'Да, отправьте'
    assert graph.states[1]['pii_aliases'] == {
        '[EMAIL_1]': ['EMAIL', email]
    }
    assert graph.resolved == [('[EMAIL_1]', email)]
    assert email not in str(
        [message.content for message in graph.checkpoint_values['messages']]
    )


async def test_different_emails_keep_distinct_aliases_across_turns():
    first_email = 'first@example.com'
    second_email = 'second@example.com'
    graph = _SessionPiiGraph(
        resolve_on_call={2: ['[EMAIL_1]', '[EMAIL_2]']}
    )
    consumer = _build_consumer(graph, _TokenSinkRecorder())

    for request_id, email in (
        ('request-1', first_email),
        ('request-2', second_email),
    ):
        payload = AgentRequestMessage(
            session_id='session-1',
            request_id=request_id,
            user_id='user-1',
            message=f'Отправьте на {email}',
        )
        _ = [
            chunk
            async for chunk in consumer._stream_answer(
                payload,
                TurnPersistenceData(),
            )
        ]

    assert graph.states[0]['messages'][0].content == 'Отправьте на [EMAIL_1]'
    assert graph.states[1]['messages'][0].content == 'Отправьте на [EMAIL_2]'
    assert graph.resolved == [
        ('[EMAIL_1]', first_email),
        ('[EMAIL_2]', second_email),
    ]


async def test_legacy_alias_without_mapping_is_reserved_and_never_rebound():
    new_email = 'new@example.com'
    graph = _SessionPiiGraph(
        checkpoint_values={
            'session_id': 'session-1',
            'messages': [HumanMessage(content='Старая почта [EMAIL_1]')],
        },
        resolve_on_call={1: ['[EMAIL_1]', '[EMAIL_2]']},
    )
    consumer = _build_consumer(graph, _TokenSinkRecorder())
    payload = AgentRequestMessage(
        session_id='session-1',
        request_id='request-2',
        user_id='user-1',
        message=f'Используйте новую почту {new_email}',
    )

    _ = [
        chunk
        async for chunk in consumer._stream_answer(
            payload,
            TurnPersistenceData(),
        )
    ]

    assert graph.states[0]['messages'][0].content == (
        'Используйте новую почту [EMAIL_2]'
    )
    assert graph.states[0]['pii_aliases'] == {
        '[EMAIL_2]': ['EMAIL', new_email]
    }
    assert graph.resolved == [
        ('[EMAIL_1]', None),
        ('[EMAIL_2]', new_email),
    ]


async def test_raw_legacy_email_is_registered_before_new_turn():
    legacy_email = 'legacy@example.com'
    graph = _SessionPiiGraph(
        checkpoint_values={
            'session_id': 'session-1',
            'messages': [
                HumanMessage(content=f'Отправьте на {legacy_email}')
            ],
        },
        resolve_on_call={1: ['[EMAIL_1]']},
    )
    consumer = _build_consumer(graph, _TokenSinkRecorder())
    payload = AgentRequestMessage(
        session_id='session-1',
        request_id='request-2',
        user_id='user-1',
        message='Да, отправьте',
    )

    _ = [
        chunk
        async for chunk in consumer._stream_answer(
            payload,
            TurnPersistenceData(),
        )
    ]

    assert graph.states[0]['pii_aliases'] == {
        '[EMAIL_1]': ['EMAIL', legacy_email]
    }
    assert graph.resolved == [('[EMAIL_1]', legacy_email)]


async def test_real_checkpointer_restores_email_alias_for_next_turn():
    email = 'checkpoint@example.com'
    resolved: list[str] = []

    def generate_direct(state: AgentState) -> dict:
        if state['messages'][-1].content == 'Да, отправьте':
            resolved.append(resolve_email_for_tool('[EMAIL_1]'))
        return {'messages': [AIMessage(content='Готово')]}

    builder = StateGraph(AgentState)
    builder.add_node('generate_direct', generate_direct)
    builder.set_entry_point('generate_direct')
    builder.set_finish_point('generate_direct')
    graph = builder.compile(checkpointer=InMemorySaver())
    consumer = AgentRequestConsumer(
        connection_url='amqp://unused',
        queue_name='agent.requests',
        dlq_name='agent.requests.dlq',
        graph=graph,
        token_sink=_TokenSinkRecorder(),
    )

    for request_id, message in (
        ('request-1', f'Отправьте на {email}'),
        ('request-2', 'Да, отправьте'),
    ):
        payload = AgentRequestMessage(
            session_id='session-with-checkpoint',
            request_id=request_id,
            user_id='user-1',
            message=message,
        )
        answer = ''.join(
            [
                chunk
                async for chunk in consumer._stream_answer(
                    payload,
                    TurnPersistenceData(),
                )
            ]
        )
        assert answer == 'Готово'

    snapshot = await graph.aget_state(
        {'configurable': {'thread_id': 'session-with-checkpoint'}}
    )
    assert resolved == [email]
    assert snapshot.values['pii_aliases'] == {
        '[EMAIL_1]': ['EMAIL', email]
    }
    human_contents = [
        message.content
        for message in snapshot.values['messages']
        if isinstance(message, HumanMessage)
    ]
    assert human_contents == ['Отправьте на [EMAIL_1]', 'Да, отправьте']


async def test_consumer_redacts_lowercase_name_after_context_marker():
    graph = _FakeGraph([[_token_event('Ответ')]])
    sink = _TokenSinkRecorder()
    consumer = _build_consumer(graph, sink)
    message = _FakeMessage(
        body=json.dumps(
            {
                'session_id': 'conversation-1',
                'request_id': 'request-1',
                'user_id': 'u1',
                'message': (
                    'меня зовут алексей петров помогите с увольнением'
                ),
            },
            ensure_ascii=False,
        ).encode(),
    )

    await consumer._handle_message(message)

    assert graph.states[0]['messages'][0].content == (
        'меня зовут [ФИО_1] помогите с увольнением'
    )


async def test_consumer_redacts_name_after_self_identification_aside():
    graph = _FakeGraph([[_token_event('Ответ')]])
    sink = _TokenSinkRecorder()
    consumer = _build_consumer(graph, sink)
    message = _FakeMessage(
        body=json.dumps(
            {
                'session_id': 'conversation-1',
                'request_id': 'request-1',
                'user_id': 'u1',
                'message': (
                    'ок, спасибо! я, кстати, Кирилл инвалид по зрению'
                ),
            },
            ensure_ascii=False,
        ).encode(),
    )

    await consumer._handle_message(message)

    assert graph.states[0]['messages'][0].content == (
        'ок, спасибо! я, кстати, [ФИО_1] инвалид по зрению'
    )


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
    message = _FakeMessage(
        body=b'{"session_id": "s1", "request_id": "r1", "user_id": "u1", "message": "?"}'
    )

    await consumer._handle_message(message)

    assert message.acked
    assert sink.calls == [
        ('r1', {'type': 'token', 'content': 'Финальный ответ.'}),
        ('r1', {'type': 'done', 'used_knowledge_base': False}),
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
    message = _FakeMessage(
        body=b'{"session_id": "s1", "request_id": "r1", "user_id": "u1", "message": "?"}'
    )

    await consumer._handle_message(message)

    assert message.acked
    assert sink.calls == [
        ('r1', {'type': 'token', 'content': 'Финальный ответ.'}),
        ('r1', {'type': 'done', 'used_knowledge_base': False}),
    ]


async def test_streams_code_authored_final_message_when_model_was_not_called():
    graph = _FakeGraph(
        [
            [
                {
                    'event': 'on_chain_end',
                    'parent_ids': [],
                    'data': {
                        'output': {
                            'messages': [
                                HumanMessage(content='send'),
                                AIMessage(content='Безопасный ответ без вызова модели.'),
                            ],
                            'retrieved_chunks': [],
                            'tool_calls': [],
                        }
                    },
                }
            ]
        ]
    )
    sink = _TokenSinkRecorder()
    consumer = _build_consumer(graph, sink)
    message = _FakeMessage(
        body=b'{"session_id": "s1", "request_id": "r1", "user_id": "u1", "message": "send"}'
    )

    await consumer._handle_message(message)

    assert message.acked
    assert sink.calls == [
        ('r1', {'type': 'token', 'content': 'Безопасный ответ без вызова модели.'}),
        ('r1', {'type': 'done', 'used_knowledge_base': False}),
    ]


async def test_stream_answer_never_emits_raw_pseudo_stream_without_safe_snapshot():
    graph = _FakeGraph(
        [
            [
                _raw_model_event('call:default_api:'),
                _raw_model_event('send_consultation_email{email=user@example.com}'),
            ]
        ]
    )
    consumer = _build_consumer(graph, _TokenSinkRecorder())
    payload = AgentRequestMessage(
        session_id='session-1',
        request_id='request-1',
        user_id='user-1',
        message='Повтори отправку',
    )

    answer = ''.join(
        [chunk async for chunk in consumer._stream_answer(payload, TurnPersistenceData())]
    )

    assert 'call:default_api:' not in answer
    assert answer == ''


async def test_confirmed_final_node_raw_stream_is_ignored_in_favor_of_safe_snapshot():
    incident = 'The user is asking: internal analysis. Rules check: allowed.'
    graph = _FakeGraph(
        [
            [
                _raw_model_event(incident, node='generate_direct'),
                _token_event('Только проверенный ответ.', node='generate_direct'),
            ]
        ]
    )
    consumer = _build_consumer(graph, _TokenSinkRecorder())
    payload = AgentRequestMessage(
        session_id='session-1',
        request_id='request-1',
        user_id='user-1',
        message='Вопрос',
    )

    answer = ''.join(
        [chunk async for chunk in consumer._stream_answer(payload, TurnPersistenceData())]
    )

    assert answer == 'Только проверенный ответ.'
    assert incident not in answer


async def test_unsafe_final_snapshot_is_not_emitted_as_sse():
    graph = _FakeGraph(
        [[_token_event('The user is asking: secret. Rules check: allowed.')]]
    )
    consumer = _build_consumer(graph, _TokenSinkRecorder())
    payload = AgentRequestMessage(
        session_id='session-1',
        request_id='request-1',
        user_id='user-1',
        message='Вопрос',
    )

    chunks = [
        chunk
        async for chunk in consumer._stream_answer(payload, TurnPersistenceData())
    ]

    assert chunks == []


@pytest.mark.parametrize(
    'unsafe_content',
    [
        '   \n\t',
        'Ответ\x00со скрытым управляющим символом',
        [{'type': 'reasoning.text', 'text': 'Скрытый разбор'}],
    ],
)
async def test_empty_controlled_or_typed_final_snapshot_is_not_emitted(
    unsafe_content,
):
    graph = _FakeGraph([[_token_event(unsafe_content)]])
    consumer = _build_consumer(graph, _TokenSinkRecorder())
    payload = AgentRequestMessage(
        session_id='session-1',
        request_id='request-1',
        user_id='user-1',
        message='Вопрос',
    )

    chunks = [
        chunk
        async for chunk in consumer._stream_answer(payload, TurnPersistenceData())
    ]

    assert chunks == []


async def test_authoritative_unsafe_root_snapshot_clears_safe_node_answer():
    root_event = {
        'event': 'on_chain_end',
        'parent_ids': [],
        'data': {
            'output': {
                'messages': [
                    HumanMessage(content='Вопрос'),
                    AIMessage(content='   '),
                ]
            }
        },
    }
    graph = _FakeGraph(
        [[_token_event('Промежуточный безопасный snapshot.'), root_event]]
    )
    consumer = _build_consumer(graph, _TokenSinkRecorder())
    payload = AgentRequestMessage(
        session_id='session-1',
        request_id='request-1',
        user_id='user-1',
        message='Вопрос',
    )

    chunks = [
        chunk
        async for chunk in consumer._stream_answer(payload, TurnPersistenceData())
    ]

    assert chunks == []


async def test_malformed_graph_events_are_ignored_before_safe_snapshot():
    graph = _FakeGraph(
        [
            [
                None,
                {
                    'event': 'on_chain_end',
                    'parent_ids': [],
                    'metadata': None,
                    'data': None,
                },
                {
                    'event': 'on_chain_end',
                    'parent_ids': ['graph-run'],
                    'metadata': {'langgraph_node': ['invalid']},
                    'data': 'invalid',
                },
                _token_event('Проверенный ответ.'),
            ]
        ]
    )
    consumer = _build_consumer(graph, _TokenSinkRecorder())
    payload = AgentRequestMessage(
        session_id='session-1',
        request_id='request-1',
        user_id='user-1',
        message='Вопрос',
    )

    answer = ''.join(
        [chunk async for chunk in consumer._stream_answer(payload, TurnPersistenceData())]
    )

    assert answer == 'Проверенный ответ.'


async def test_stream_answer_uses_final_node_snapshot_with_parent_ids():
    graph = _FakeGraph(
        [
            [
                {
                    'event': 'on_chain_end',
                    'parent_ids': ['graph-run'],
                    'metadata': {'langgraph_node': 'generate_direct'},
                    'data': {
                        'output': {
                            'messages': [
                                HumanMessage(content='Повтори отправку'),
                                AIMessage(content='Безопасный ответ из final node.'),
                            ]
                        }
                    },
                }
            ]
        ]
    )
    consumer = _build_consumer(graph, _TokenSinkRecorder())
    payload = AgentRequestMessage(
        session_id='session-1',
        request_id='request-1',
        user_id='user-1',
        message='Повтори отправку',
    )

    answer = ''.join(
        [chunk async for chunk in consumer._stream_answer(payload, TurnPersistenceData())]
    )

    assert answer == 'Безопасный ответ из final node.'


async def test_empty_llm_stream_is_terminal_without_replaying_graph():
    graph = _FakeGraph([[EmptyLlmStreamError()]])
    sink = _TokenSinkRecorder()
    consumer = _build_consumer(graph, sink, retries=3)
    message = _FakeMessage(
        body=b'{"session_id": "s1", "request_id": "r1", "user_id": "u1", "message": "?"}'
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
                'detail': 'Сервис временно недоступен, попробуйте позже.',
            },
        )
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
    message = _FakeMessage(
        body=b'{"session_id": "s1", "request_id": "r1", "user_id": "u1", "message": "?"}'
    )

    await consumer._handle_message(message)

    assert graph.call_count == 2
    assert message.acked
    assert ('r1', {'type': 'done', 'used_knowledge_base': False}) in sink.calls
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
    message = _FakeMessage(
        body=b'{"session_id": "s1", "request_id": "r1", "user_id": "u1", "message": "?"}'
    )

    await consumer._handle_message(message)

    assert graph.call_count == 3
    assert message.nacked_requeue is False
    assert not message.acked
    assert sink.calls[-1] == ('r1', {'type': 'error', 'detail': 'Сервис временно недоступен, попробуйте позже.'})


async def test_raw_model_chunk_does_not_start_sse_and_graph_can_retry_safely():
    graph = _FakeGraph(
        [
            [
                _raw_model_event('Сырой небезопасный префикс'),
                RuntimeError('обрыв соединения с LLM'),
            ],
            [_token_event('Полный проверенный ответ')],
        ]
    )
    sink = _TokenSinkRecorder()
    consumer = _build_consumer(graph, sink, retries=3)
    message = _FakeMessage(
        body=b'{"session_id": "s1", "request_id": "r1", "user_id": "u1", "message": "?"}'
    )

    await consumer._handle_message(message)

    assert graph.call_count == 2
    assert message.acked
    assert message.nacked_requeue is None
    assert sink.calls == [
        ('r1', {'type': 'token', 'content': 'Полный проверенный ответ'}),
        ('r1', {'type': 'done', 'used_knowledge_base': False}),
    ]


async def test_failure_after_mutating_tool_call_never_retries_and_acks():
    """Даже без отправленного SSE-токена весь граф нельзя повторять после
    вызова email-тулы: письмо могло быть принято до последующего сбоя."""
    graph = _MutatingFailureGraph()
    sink = _TokenSinkRecorder()
    consumer = _build_consumer(graph, sink, retries=3)
    message = _FakeMessage(
        body=b'{"session_id": "s1", "request_id": "r1", "user_id": "u1", "message": "send"}'
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


async def test_successful_message_persists_question_answer_and_sources():
    graph = _FakeGraph(
        [
            [
                {
                    'event': 'on_chain_end',
                    'data': {
                        'output': {
                            'retrieved_chunks': [{'document_id': 'doc-1'}],
                            'tool_calls': ['vera_rag_kb'],
                        }
                    },
                },
                _token_event('Полный ответ'),
            ]
        ]
    )
    sink = _TokenSinkRecorder()
    persistence_service = AsyncMock(spec=ChatPersistenceService)
    persistence_service.start_turn.return_value = TurnStartResult(
        outcome=START_CLAIMED,
        status=STATUS_PROCESSING,
    )

    @asynccontextmanager
    async def persistence_factory():
        yield persistence_service

    consumer = AgentRequestConsumer(
        connection_url='amqp://unused',
        queue_name='agent.requests',
        dlq_name='agent.requests.dlq',
        graph=graph,
        token_sink=sink,
        persistence_service_factory=persistence_factory,
    )
    message = _FakeMessage(
        body=b'{"session_id": "s1", "request_id": "r1", "user_id": "u1", "message": "question"}'
    )

    await consumer._handle_message(message)

    start_call = persistence_service.start_turn.await_args.kwargs
    assert start_call['session_id'] == 's1'
    assert start_call['request_id'] == 'r1'
    assert start_call['user_id'] == 'u1'
    assert start_call['anonymous_token_hash'] is None
    assert start_call['question'] == 'question'
    # Аренда обязана уходить в persistence: без неё брошенная реплика
    # никогда не будет перезахвачена после сбоя.
    assert start_call['worker_id']
    assert start_call['lease_seconds'] > 0
    complete_call = persistence_service.complete_turn.await_args.kwargs
    assert complete_call['answer'] == 'Полный ответ'
    assert complete_call['sources'] == [{'document_id': 'doc-1'}]
    assert complete_call['technical_metadata']['tool_calls'] == ['vera_rag_kb']
    sse_answer = ''.join(
        event['content']
        for _request_id, event in sink.calls
        if event.get('type') == 'token'
    )
    final_ai_message = graph._events_per_call[0][-1]['data']['output']['messages'][0]
    assert sse_answer == final_ai_message.content == complete_call['answer']
    assert message.acked is True


async def test_stream_answer_uses_only_root_graph_metadata():
    child_sources = [{'document_id': 'child'}]
    authoritative_sources = [{'document_id': 'root'}]
    graph = _FakeGraph(
        [
            [
                {
                    'event': 'on_chain_end',
                    'parent_ids': ['root-run'],
                    'data': {
                        'output': {
                            'retrieved_chunks': child_sources,
                            'tool_calls': ['vera_rag_kb'],
                        }
                    },
                },
                {
                    'event': 'on_chain_end',
                    'parent_ids': [],
                    'data': {
                        'output': {
                            'retrieved_chunks': authoritative_sources,
                            'tool_calls': [
                                'vera_rag_kb',
                                'vera_rag_kb',
                                'send_consultation_email',
                            ],
                        }
                    },
                },
                _token_event('Ответ'),
            ]
        ]
    )
    sink = _TokenSinkRecorder()
    consumer = _build_consumer(graph, sink)
    payload = AgentRequestMessage(
        session_id='session-1',
        request_id='request-1',
        user_id='user-1',
        message='Вопрос',
    )
    persistence_data = TurnPersistenceData()

    answer = ''.join(
        [
            chunk
            async for chunk in consumer._stream_answer(
                payload,
                persistence_data,
            )
        ]
    )

    assert answer == 'Ответ'
    assert persistence_data.sources == authoritative_sources
    assert persistence_data.tool_calls == [
        'vera_rag_kb',
        'send_consultation_email',
    ]


def _root_snapshot_event(retrieved_chunks: list, tool_calls: list[str]) -> dict:
    """Корневой graph output — единственный authoritative snapshot, из
    которого consumer берёт чанки и имена вызванных тулов."""
    return {
        'event': 'on_chain_end',
        'parent_ids': [],
        'data': {
            'output': {
                'messages': [],
                'retrieved_chunks': retrieved_chunks,
                'tool_calls': tool_calls,
            }
        },
    }


async def test_done_reports_knowledge_base_use_when_chunks_were_retrieved():
    """Найденные чанки — признак, по которому сайт показывает под ответом
    кнопку «Объяснить проще»."""
    graph = _FakeGraph(
        [
            [
                _token_event('Квота составляет 2%.', node='generate_with_context'),
                _root_snapshot_event(
                    [{'chunk_id': 'c1', 'text': 'Норма о квоте'}],
                    ['vera_rag_kb'],
                ),
            ]
        ]
    )
    sink = _TokenSinkRecorder()
    consumer = _build_consumer(graph, sink)
    message = _FakeMessage(
        body=b'{"session_id": "s1", "request_id": "r1", "user_id": "u1", "message": "?"}'
    )

    await consumer._handle_message(message)

    assert message.acked
    assert sink.calls[-1] == ('r1', {'type': 'done', 'used_knowledge_base': True})


async def test_done_reports_no_knowledge_base_use_when_search_found_nothing():
    """Честный отказ «в базе знаний ничего не нашлось» не даёт кнопку:
    упрощать нечего, хотя поиск и вызывался."""
    graph = _FakeGraph(
        [
            [
                _token_event(
                    'Не нашёл ответа, попробуйте переформулировать.',
                    node='generate_with_context',
                ),
                _root_snapshot_event([], ['vera_rag_kb']),
            ]
        ]
    )
    sink = _TokenSinkRecorder()
    consumer = _build_consumer(graph, sink)
    message = _FakeMessage(
        body=b'{"session_id": "s1", "request_id": "r1", "user_id": "u1", "message": "?"}'
    )

    await consumer._handle_message(message)

    assert message.acked
    assert sink.calls[-1] == ('r1', {'type': 'done', 'used_knowledge_base': False})
