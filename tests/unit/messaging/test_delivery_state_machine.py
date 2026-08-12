"""Проверки контракта доставки: по одному тесту на строку таблицы исходов.

Инварианты, которые обязаны выполняться в каждом сценарии:

- ровно один `ack` или `nack` на доставку;
- ровно одно терминальное SSE-событие на адресуемый запрос;
- `done` отправляется только после подтверждённого сохранения;
- повторная доставка воспроизводит сохранённый исход, а не выдаёт любой
  сохранённый текст за успех.
"""

import asyncio
import json
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.db.models.chat_turn import (
    STATUS_COMPLETED,
    STATUS_DELIVERY_UNCONFIRMED,
    STATUS_GENERATION_FAILED,
    STATUS_PROCESSING,
    STATUS_STREAM_INTERRUPTED,
)
from app.exceptions.chat_turn import ChatPersistenceServiceError
from app.messaging.consumer import (
    COMMIT_FAILED_MESSAGE,
    DUPLICATE_IN_PROGRESS_MESSAGE,
    GENERATION_FAILED_MESSAGE,
    MUTATING_TOOL_UNCONFIRMED_MESSAGE,
    SHUTDOWN_MESSAGE,
    STREAM_INTERRUPTED_MESSAGE,
    AgentRequestConsumer,
)
from app.observability.request_trace import get_request_trace
from app.services.chat_persistence import (
    START_CLAIMED,
    START_DUPLICATE_IN_PROGRESS,
    START_DUPLICATE_TERMINAL,
    ChatPersistenceService,
    TurnStartResult,
)

TERMINAL_TYPES = frozenset({'done', 'error'})


class _FakeMessage:
    def __init__(self, body: bytes):
        self.body = body
        self.ack_count = 0
        self.nack_count = 0
        self.nacked_requeue: bool | None = None

    async def ack(self):
        self.ack_count += 1

    async def nack(self, requeue: bool = True):
        self.nack_count += 1
        self.nacked_requeue = requeue

    @property
    def settlements(self) -> int:
        return self.ack_count + self.nack_count


class _TokenSinkRecorder:
    def __init__(self):
        self.calls: list[tuple[str, dict]] = []

    async def __call__(self, request_id: str, event: dict) -> None:
        self.calls.append((request_id, event))

    @property
    def events(self) -> list[dict]:
        return [event for _, event in self.calls]

    @property
    def terminal_events(self) -> list[dict]:
        return [event for event in self.events if event.get('type') in TERMINAL_TYPES]


class _ScriptedGraph:
    """Отдаёт заранее заданный поток событий на каждый вызов графа."""

    def __init__(self, events_per_call: list[list]):
        self._events_per_call = events_per_call
        self.call_count = 0

    def astream_events(self, state, config, version='v2'):
        events = self._events_per_call[min(self.call_count, len(self._events_per_call) - 1)]
        self.call_count += 1

        async def _generator():
            for item in events:
                if isinstance(item, BaseException):
                    raise item
                yield item

        return _generator()


class _MutatingFailureGraph:
    def astream_events(self, state, config, version='v2'):
        async def _generator():
            trace_data = get_request_trace()
            trace_data.mutating_tool_called = True
            raise RuntimeError('сбой после начала отправки консультации')
            yield

        return _generator()


def _token_event(content: str) -> dict:
    return {
        'event': 'on_chat_model_stream',
        'metadata': {'langgraph_node': 'generate_direct'},
        'data': {'chunk': SimpleNamespace(content=content)},
    }


def _payload(request_id: str = 'r1') -> bytes:
    return json.dumps(
        {
            'session_id': 's1',
            'request_id': request_id,
            'user_id': 'user@example.com',
            'anonymous_token_hash': None,
            'message': 'Вопрос',
        }
    ).encode()


def _build_consumer(graph, sink, persistence_service=None, retries: int = 3):
    factory = None
    if persistence_service is not None:

        @asynccontextmanager
        async def factory():
            yield persistence_service

    return AgentRequestConsumer(
        connection_url='amqp://unused',
        queue_name='agent.requests',
        dlq_name='agent.requests.dlq',
        graph=graph,
        token_sink=sink,
        persistence_service_factory=factory,
        retries=retries,
        persistence_retries=2,
        worker_id='worker-test',
    )


def _persistence(outcome: str = START_CLAIMED, **kwargs) -> AsyncMock:
    service = AsyncMock(spec=ChatPersistenceService)
    service.start_turn.return_value = TurnStartResult(
        outcome=outcome,
        status=kwargs.pop('status', STATUS_PROCESSING),
        **kwargs,
    )
    return service


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    async def _instant_sleep(_seconds):
        return None

    monkeypatch.setattr('app.messaging.consumer.asyncio.sleep', _instant_sleep)


async def test_done_is_sent_only_after_successful_commit():
    """Успех: `complete_turn` предшествует `done`, доставка подтверждается."""
    graph = _ScriptedGraph([[_token_event('Ответ')]])
    sink = _TokenSinkRecorder()
    service = _persistence()
    consumer = _build_consumer(graph, sink, service)
    message = _FakeMessage(_payload())

    await consumer._handle_message(message)

    service.complete_turn.assert_awaited_once()
    assert sink.terminal_events == [{'type': 'done', 'used_knowledge_base': False}]
    assert message.ack_count == 1
    assert message.settlements == 1


async def test_failed_commit_reports_error_instead_of_done():
    """Сбой сохранения после стриминга не маскируется под успех (VERA-004)."""
    graph = _ScriptedGraph([[_token_event('Ответ')]])
    sink = _TokenSinkRecorder()
    service = _persistence()
    service.complete_turn.side_effect = ChatPersistenceServiceError
    consumer = _build_consumer(graph, sink, service)
    message = _FakeMessage(_payload())

    await consumer._handle_message(message)

    assert sink.terminal_events == [{'type': 'error', 'detail': COMMIT_FAILED_MESSAGE}]
    fail_call = service.fail_turn.await_args.kwargs
    assert fail_call['status'] == STATUS_DELIVERY_UNCONFIRMED
    assert fail_call['answer'] == 'Ответ'
    assert fail_call['terminal_detail'] == COMMIT_FAILED_MESSAGE
    assert message.ack_count == 1
    assert message.settlements == 1


async def test_empty_stream_is_content_error_not_done():
    """Пустой поток модели — ошибка ответа, а не успех (VERA-018)."""
    graph = _ScriptedGraph([[], [], []])
    sink = _TokenSinkRecorder()
    service = _persistence()
    consumer = _build_consumer(graph, sink, service)
    message = _FakeMessage(_payload())

    await consumer._handle_message(message)

    assert sink.terminal_events == [
        {'type': 'error', 'detail': GENERATION_FAILED_MESSAGE}
    ]
    service.complete_turn.assert_not_awaited()
    assert service.fail_turn.await_args.kwargs['status'] == STATUS_GENERATION_FAILED
    # Пустой ответ уже получил локальные retries в `astream_tokens`; повтор
    # всего графа и повторная доставка через DLQ только заново запускают тот
    # же плохой LLM-ответ. Клиент получает терминальную ошибку, сообщение
    # подтверждается, а пользователь может повторить запрос новой репликой.
    assert message.ack_count == 1
    assert message.nacked_requeue is None
    assert message.settlements == 1


async def test_failure_before_first_token_goes_to_dlq_as_generation_failed():
    graph = _ScriptedGraph([[RuntimeError('LLM down')]])
    sink = _TokenSinkRecorder()
    service = _persistence()
    consumer = _build_consumer(graph, sink, service, retries=2)
    message = _FakeMessage(_payload())

    await consumer._handle_message(message)

    assert sink.terminal_events == [
        {'type': 'error', 'detail': GENERATION_FAILED_MESSAGE}
    ]
    assert service.fail_turn.await_args.kwargs['status'] == STATUS_GENERATION_FAILED
    assert message.nacked_requeue is False
    assert message.settlements == 1


async def test_failure_after_first_token_is_stream_interrupted_and_acked():
    graph = _ScriptedGraph([[_token_event('Часть'), RuntimeError('обрыв')]])
    sink = _TokenSinkRecorder()
    service = _persistence()
    consumer = _build_consumer(graph, sink, service)
    message = _FakeMessage(_payload())

    await consumer._handle_message(message)

    assert sink.terminal_events == [
        {'type': 'error', 'detail': STREAM_INTERRUPTED_MESSAGE}
    ]
    fail_call = service.fail_turn.await_args.kwargs
    assert fail_call['status'] == STATUS_STREAM_INTERRUPTED
    assert fail_call['answer'] == 'Часть'
    # Повторная обработка показала бы пользователю ответ дважды.
    assert message.ack_count == 1
    assert message.settlements == 1


async def test_failure_after_mutating_tool_is_delivery_unconfirmed():
    sink = _TokenSinkRecorder()
    service = _persistence()
    consumer = _build_consumer(_MutatingFailureGraph(), sink, service)
    message = _FakeMessage(_payload())

    await consumer._handle_message(message)

    assert sink.terminal_events == [
        {'type': 'error', 'detail': MUTATING_TOOL_UNCONFIRMED_MESSAGE}
    ]
    assert service.fail_turn.await_args.kwargs['status'] == STATUS_DELIVERY_UNCONFIRMED
    assert message.ack_count == 1


async def test_duplicate_with_live_lease_is_reported_as_in_progress():
    graph = _ScriptedGraph([[_token_event('не должен выполниться')]])
    sink = _TokenSinkRecorder()
    service = _persistence(START_DUPLICATE_IN_PROGRESS)
    consumer = _build_consumer(graph, sink, service)
    message = _FakeMessage(_payload())

    await consumer._handle_message(message)

    assert sink.terminal_events == [
        {'type': 'error', 'detail': DUPLICATE_IN_PROGRESS_MESSAGE}
    ]
    assert graph.call_count == 0
    assert message.ack_count == 1


async def test_duplicate_completed_replays_saved_answer_and_done():
    sink = _TokenSinkRecorder()
    service = _persistence(
        START_DUPLICATE_TERMINAL,
        status=STATUS_COMPLETED,
        answer='Сохранённый ответ',
    )
    graph = _ScriptedGraph([[]])
    consumer = _build_consumer(graph, sink, service)
    message = _FakeMessage(_payload())

    await consumer._handle_message(message)

    assert sink.events == [
        {'type': 'token', 'content': 'Сохранённый ответ'},
        {'type': 'done', 'used_knowledge_base': False},
    ]
    assert graph.call_count == 0
    assert message.ack_count == 1


async def test_duplicate_delivery_unconfirmed_replays_error_not_done():
    """Зафиксированная ошибка не превращается в успех при повторе (VERA-034)."""
    sink = _TokenSinkRecorder()
    service = _persistence(
        START_DUPLICATE_TERMINAL,
        status=STATUS_DELIVERY_UNCONFIRMED,
        answer='Частичный ответ',
        terminal_detail=COMMIT_FAILED_MESSAGE,
    )
    graph = _ScriptedGraph([[]])
    consumer = _build_consumer(graph, sink, service)
    message = _FakeMessage(_payload())

    await consumer._handle_message(message)

    assert sink.terminal_events == [{'type': 'error', 'detail': COMMIT_FAILED_MESSAGE}]
    assert not any(event.get('type') == 'done' for event in sink.events)
    assert graph.call_count == 0
    assert message.ack_count == 1


async def test_invalid_payload_goes_to_dlq_without_terminal_event():
    """Поток по непроверенному `request_id` завершать нельзя (VERA-016)."""
    sink = _TokenSinkRecorder()
    consumer = _build_consumer(_ScriptedGraph([[]]), sink)
    message = _FakeMessage(b'{"session_id": "s1"}')

    await consumer._handle_message(message)

    assert sink.calls == []
    assert message.nacked_requeue is False
    assert message.settlements == 1


async def test_transient_start_persistence_error_is_retried():
    """Временная недоступность БД до графа не должна терять запрос."""
    graph = _ScriptedGraph([[_token_event('Ответ')]])
    sink = _TokenSinkRecorder()
    service = _persistence()
    service.start_turn.side_effect = [
        ChatPersistenceServiceError,
        TurnStartResult(outcome=START_CLAIMED, status=STATUS_PROCESSING),
    ]
    consumer = _build_consumer(graph, sink, service)
    message = _FakeMessage(_payload())

    await consumer._handle_message(message)

    assert service.start_turn.await_count == 2
    assert sink.terminal_events == [{'type': 'done', 'used_knowledge_base': False}]
    assert message.ack_count == 1


async def test_persistent_start_failure_reports_error_once_and_settles():
    graph = _ScriptedGraph([[_token_event('Ответ')]])
    sink = _TokenSinkRecorder()
    service = _persistence()
    service.start_turn.side_effect = ChatPersistenceServiceError
    consumer = _build_consumer(graph, sink, service)
    message = _FakeMessage(_payload())

    await consumer._handle_message(message)

    assert len(sink.terminal_events) == 1
    assert sink.terminal_events[0]['type'] == 'error'
    assert graph.call_count == 0
    assert message.settlements == 1


async def test_unexpected_error_still_sends_terminal_event_and_settles():
    """Никакая непредусмотренная ошибка не оставляет delivery неподтверждённой."""
    graph = _ScriptedGraph([[_token_event('Ответ')]])
    sink = _TokenSinkRecorder()
    service = _persistence()
    service.start_turn.side_effect = ValueError('неожиданная ошибка')
    consumer = _build_consumer(graph, sink, service)
    message = _FakeMessage(_payload())

    await consumer._handle_message(message)

    assert len(sink.terminal_events) == 1
    assert sink.terminal_events[0]['type'] == 'error'
    assert message.settlements == 1


async def test_shutdown_records_unconfirmed_outcome_and_propagates():
    """Остановка сервиса фиксирует неопределённый исход и не глотает отмену."""
    graph = _ScriptedGraph([[_token_event('Часть'), asyncio.CancelledError()]])
    sink = _TokenSinkRecorder()
    service = _persistence()
    consumer = _build_consumer(graph, sink, service)
    message = _FakeMessage(_payload())

    with pytest.raises(asyncio.CancelledError):
        await consumer._handle_message(message)

    assert sink.terminal_events == [{'type': 'error', 'detail': SHUTDOWN_MESSAGE}]
    assert service.fail_turn.await_args.kwargs['status'] == STATUS_DELIVERY_UNCONFIRMED
    assert message.settlements == 1


async def test_duplicate_completed_replays_knowledge_base_flag():
    """Повтор доставки обязан воспроизвести и признак использования базы
    знаний: иначе после дубликата живой клиент потеряет кнопку «Объяснить
    проще», которая осталась бы в истории."""
    sink = _TokenSinkRecorder()
    service = _persistence(
        START_DUPLICATE_TERMINAL,
        status=STATUS_COMPLETED,
        answer='Сохранённый ответ',
        used_knowledge_base=True,
    )
    graph = _ScriptedGraph([[]])
    consumer = _build_consumer(graph, sink, service)
    message = _FakeMessage(_payload())

    await consumer._handle_message(message)

    assert sink.terminal_events == [{'type': 'done', 'used_knowledge_base': True}]
    assert graph.call_count == 0
