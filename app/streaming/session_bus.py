import asyncio
import logging
import time
from collections import deque
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass, field

from app.exceptions.streaming import SessionAlreadySubscribedError

logger = logging.getLogger('vera_agent_service')

LATE_CONNECT_BUFFER_SECONDS: float = 60.0
SUBSCRIBER_QUEUE_MAX_EVENTS: int = 256
LATE_CONNECT_BUFFER_MAX_EVENTS: int = 256
LATE_CONNECT_BUFFER_MAX_REQUESTS: int = 1024
REQUEST_STATE_MAX_ENTRIES: int = 2048
BUFFER_CLEANUP_INTERVAL_SECONDS: float = 15.0
REQUEST_DEADLINE_SECONDS: float = 420.0
SLOW_SUBSCRIBER_DETAIL = 'Поток ответа закрыт: клиент не успевает принимать данные.'
LATE_BUFFER_OVERFLOW_DETAIL = 'Поток ответа закрыт: буфер событий переполнен.'
LATE_BUFFER_EXPIRED_DETAIL = 'Поток ответа закрыт: срок хранения буфера истёк.'
REQUEST_DEADLINE_DETAIL = 'Время ожидания ответа истекло. Ответ появится в истории.'


class SessionBusCapacityExceededError(Exception):
    """In-memory state заполнен и не может безопасно забыть живой запрос."""


class SessionReplayUnavailableError(Exception):
    """Запрошенный SSE id уже вышел из bounded in-memory replay window."""


class SessionReplayCompleteError(Exception):
    """Клиент уже подтвердил последний terminal event этого request."""


@dataclass(frozen=True)
class SequencedStreamEvent:
    """Событие вместе с возрастающим SSE id конкретного request_id."""

    event_id: int
    payload: dict
    completes_request: bool = False


@dataclass
class _RequestStreamState:
    started_at: float
    deadline_at: float
    last_activity: float
    last_event_id: int = 0
    replay_floor_event_id: int = 0
    queue: asyncio.Queue[SequencedStreamEvent] | None = None
    buffer: deque[tuple[float, SequencedStreamEvent]] = field(
        default_factory=deque,
    )
    terminal_event: SequencedStreamEvent | None = None
    terminal_at: float | None = None
    producer_finished: bool = False
    has_had_subscriber: bool = False


class SessionBus:
    """Ограниченный in-memory transport событий одного процесса.

    Первый terminal event закрывает внешний stream state. Если причиной был
    медленный клиент или deadline, consumer продолжает граф и persistence,
    но его последующие события становятся no-op и не создают второй terminal.
    """

    def __init__(
        self,
        buffer_seconds: float = LATE_CONNECT_BUFFER_SECONDS,
        *,
        subscriber_queue_max_events: int = SUBSCRIBER_QUEUE_MAX_EVENTS,
        buffer_max_events: int = LATE_CONNECT_BUFFER_MAX_EVENTS,
        buffer_max_requests: int = LATE_CONNECT_BUFFER_MAX_REQUESTS,
        state_max_entries: int = REQUEST_STATE_MAX_ENTRIES,
        cleanup_interval_seconds: float = BUFFER_CLEANUP_INTERVAL_SECONDS,
        request_deadline_seconds: float = REQUEST_DEADLINE_SECONDS,
        monotonic_clock: Callable[[], float] | None = None,
    ):
        if min(
            buffer_seconds,
            subscriber_queue_max_events,
            buffer_max_events,
            buffer_max_requests,
            state_max_entries,
            cleanup_interval_seconds,
            request_deadline_seconds,
        ) <= 0:
            raise ValueError('Лимиты SessionBus должны быть положительными')
        if buffer_max_events > subscriber_queue_max_events:
            raise ValueError(
                'Late-connect buffer не может быть больше subscriber queue'
            )
        if buffer_max_requests > state_max_entries:
            raise ValueError(
                'Лимит late-connect buffers не может быть больше общего state'
            )

        self._buffer_seconds = buffer_seconds
        self._subscriber_queue_max_events = subscriber_queue_max_events
        self._buffer_max_events = buffer_max_events
        self._buffer_max_requests = buffer_max_requests
        self._state_max_entries = state_max_entries
        self._cleanup_interval_seconds = cleanup_interval_seconds
        self._request_deadline_seconds = request_deadline_seconds
        self._clock = monotonic_clock or time.monotonic
        self._states: dict[str, _RequestStreamState] = {}
        self._cleanup_task: asyncio.Task[None] | None = None

    @property
    def _queues(self) -> dict[str, asyncio.Queue[SequencedStreamEvent]]:
        """Совместимое диагностическое представление активных subscriber."""
        return {
            request_id: state.queue
            for request_id, state in self._states.items()
            if state.queue is not None
        }

    @property
    def _buffers(
        self,
    ) -> dict[str, deque[tuple[float, SequencedStreamEvent]]]:
        """Совместимое диагностическое представление late-connect buffers."""
        return {
            request_id: state.buffer
            for request_id, state in self._states.items()
            if (
                state.queue is None
                and state.buffer
                and state.terminal_event is None
            )
        }

    async def start(self) -> None:
        """Запускает одну фоновую задачу очистки просроченных буферов."""
        if self._cleanup_task is not None and not self._cleanup_task.done():
            return
        self._cleanup_task = asyncio.create_task(
            self._cleanup_loop(),
            name='session-bus-buffer-cleanup',
        )

    async def stop(self) -> None:
        """Останавливает cleanup task при завершении lifespan приложения."""
        task = self._cleanup_task
        self._cleanup_task = None
        if task is None:
            return
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task

    def subscribe(
        self,
        request_id: str,
        *,
        last_event_id: int | None = None,
    ) -> asyncio.Queue[SequencedStreamEvent]:
        """Регистрирует единственного subscriber и переносит свежий buffer."""
        now = self._clock()
        self._prune_expired_state(now)
        state = self._get_or_create_state(request_id, now)
        if state.queue is not None:
            raise SessionAlreadySubscribedError(request_id)

        if state.buffer and now - state.buffer[0][0] > self._buffer_seconds:
            self._terminalize(
                request_id,
                state,
                detail=LATE_BUFFER_EXPIRED_DETAIL,
                now=now,
                producer_terminal=False,
            )

        requested_after = last_event_id if last_event_id is not None else 0
        if requested_after < 0 or requested_after > state.last_event_id:
            raise SessionReplayUnavailableError

        if (
            requested_after < state.replay_floor_event_id
            and (
                state.terminal_event is None
                or state.terminal_event.payload.get('type') == 'done'
            )
        ):
            raise SessionReplayUnavailableError

        if (
            state.terminal_event is not None
            and requested_after == state.terminal_event.event_id
        ):
            raise SessionReplayCompleteError

        # После producer terminal TTL отсчитывается от terminal_at, поэтому
        # replay либо остаётся полным, либо удаляется целиком. Нельзя отдавать
        # свежий suffix + done после истечения первого token.
        fresh_events = [
            event
            for _, event in state.buffer
            if event.event_id > requested_after
        ]

        queue: asyncio.Queue[SequencedStreamEvent] = asyncio.Queue(
            maxsize=self._subscriber_queue_max_events,
        )
        state.queue = queue
        state.last_activity = now
        state.has_had_subscriber = True
        if state.terminal_event is not None and not fresh_events:
            queue.put_nowait(state.terminal_event)
            return queue
        if len(fresh_events) > queue.maxsize:
            self._terminalize(
                request_id,
                state,
                detail=LATE_BUFFER_OVERFLOW_DETAIL,
                now=now,
                producer_terminal=False,
            )
            return queue
        for event in fresh_events:
            queue.put_nowait(event)
        if state.terminal_event is not None and not any(
            event.payload.get('type') in ('done', 'error')
            for event in fresh_events
        ):
            queue.put_nowait(state.terminal_event)
        return queue

    def unsubscribe(
        self,
        request_id: str,
        queue: asyncio.Queue[SequencedStreamEvent],
    ) -> None:
        """Освобождает subscriber-slot только для текущей очереди."""
        state = self._states.get(request_id)
        if state is not None and state.queue is queue:
            state.queue = None
            state.last_activity = self._clock()

    async def publish(self, request_id: str, event: dict) -> None:
        """Неблокирующе публикует событие либо сохраняет bounded replay."""
        now = self._clock()
        self._prune_expired_state(now)
        try:
            state = self._get_or_create_state(request_id, now)
        except SessionBusCapacityExceededError:
            logger.error(
                'SSE event запроса %s отброшен: исчерпан лимит request state',
                request_id,
            )
            return
        event_type = event.get('type')

        if state.terminal_event is not None:
            if event_type in ('done', 'error'):
                state.producer_finished = True
                state.last_activity = now
            return

        if now >= state.deadline_at:
            self._terminalize(
                request_id,
                state,
                detail=REQUEST_DEADLINE_DETAIL,
                now=now,
                producer_terminal=False,
            )
            if event_type in ('done', 'error'):
                state.producer_finished = True
            return

        if event_type in ('done', 'error'):
            if (
                state.buffer
                and now - state.buffer[0][0] > self._buffer_seconds
            ):
                self._terminalize(
                    request_id,
                    state,
                    detail=LATE_BUFFER_EXPIRED_DETAIL,
                    now=now,
                    producer_terminal=False,
                )
                state.producer_finished = True
                return
            if state.queue is not None and state.queue.full():
                self._terminalize(
                    request_id,
                    state,
                    detail=SLOW_SUBSCRIBER_DETAIL,
                    now=now,
                    producer_terminal=False,
                )
                state.producer_finished = True
                return
            if (
                state.queue is None
                and not state.has_had_subscriber
                and len(state.buffer) >= self._buffer_max_events
            ):
                self._terminalize(
                    request_id,
                    state,
                    detail=LATE_BUFFER_OVERFLOW_DETAIL,
                    now=now,
                    producer_terminal=False,
                )
                state.producer_finished = True
                return
            self._terminalize(
                request_id,
                state,
                payload=event,
                now=now,
                producer_terminal=True,
            )
            return

        queue = state.queue
        if queue is not None:
            if queue.full():
                self._terminalize(
                    request_id,
                    state,
                    detail=SLOW_SUBSCRIBER_DETAIL,
                    now=now,
                    producer_terminal=False,
                )
                logger.warning(
                    '⚠️ SSE subscriber запроса %s закрыт из-за переполнения очереди',
                    request_id,
                )
                return
            sequenced = self._sequence_event(state, event, now)
            self._remember_event(state, sequenced, now, allow_roll=True)
            queue.put_nowait(sequenced)
            return

        if (
            not state.has_had_subscriber
            and len(state.buffer) >= self._buffer_max_events
        ):
            self._terminalize(
                request_id,
                state,
                detail=LATE_BUFFER_OVERFLOW_DETAIL,
                now=now,
                producer_terminal=False,
            )
            logger.warning(
                '⚠️ Late-connect buffer запроса %s закрыт из-за переполнения',
                request_id,
            )
            return

        self._free_buffer_slot_if_needed(request_id, now)
        if state.terminal_event is None:
            sequenced = self._sequence_event(state, event, now)
            self._remember_event(
                state,
                sequenced,
                now,
                allow_roll=state.has_had_subscriber,
            )

    def remaining_deadline_seconds(self, request_id: str) -> float:
        """Возвращает остаток общего deadline, не сбрасываемого reconnect-ом."""
        now = self._clock()
        state = self._get_or_create_state(request_id, now)
        return max(0.0, state.deadline_at - now)

    def create_heartbeat(self, request_id: str) -> SequencedStreamEvent:
        """Создаёт heartbeat либо возвращает уже первый terminal event."""
        now = self._clock()
        state = self._get_or_create_state(request_id, now)
        if state.terminal_event is not None:
            return state.terminal_event
        if now >= state.deadline_at:
            return self._terminalize(
                request_id,
                state,
                detail=REQUEST_DEADLINE_DETAIL,
                now=now,
                producer_terminal=False,
            )
        return self._sequence_event(
            state,
            {'type': 'heartbeat', 'ts': int(time.time())},
            now,
        )

    def terminalize_deadline(
        self,
        request_id: str,
    ) -> SequencedStreamEvent:
        """Фиксирует первый terminal request event по общему deadline."""
        now = self._clock()
        state = self._get_or_create_state(request_id, now)
        return self._terminalize(
            request_id,
            state,
            detail=REQUEST_DEADLINE_DETAIL,
            now=now,
            producer_terminal=False,
        )

    def _get_or_create_state(
        self,
        request_id: str,
        now: float,
    ) -> _RequestStreamState:
        state = self._states.get(request_id)
        if state is None:
            if len(self._states) >= self._state_max_entries:
                raise SessionBusCapacityExceededError
            state = _RequestStreamState(
                started_at=now,
                deadline_at=now + self._request_deadline_seconds,
                last_activity=now,
            )
            self._states[request_id] = state
        return state

    @staticmethod
    def _sequence_event(
        state: _RequestStreamState,
        payload: dict,
        now: float,
        *,
        completes_request: bool = False,
    ) -> SequencedStreamEvent:
        state.last_event_id += 1
        state.last_activity = now
        return SequencedStreamEvent(
            event_id=state.last_event_id,
            payload=dict(payload),
            completes_request=completes_request,
        )

    def _terminalize(
        self,
        request_id: str,
        state: _RequestStreamState,
        *,
        now: float,
        producer_terminal: bool,
        payload: dict | None = None,
        detail: str | None = None,
    ) -> SequencedStreamEvent:
        if state.terminal_event is not None:
            if producer_terminal:
                state.producer_finished = True
                state.last_activity = now
            return state.terminal_event

        terminal_payload = payload or {'type': 'error', 'detail': detail}
        terminal = self._sequence_event(
            state,
            terminal_payload,
            now,
            completes_request=producer_terminal,
        )
        state.terminal_event = terminal
        state.terminal_at = now
        state.producer_finished = producer_terminal

        queue = state.queue
        if producer_terminal:
            self._remember_event(
                state,
                terminal,
                now,
                allow_roll=queue is not None or state.has_had_subscriber,
            )
            if queue is not None:
                queue.put_nowait(terminal)
        else:
            state.buffer.clear()
            state.replay_floor_event_id = terminal.event_id - 1
            state.buffer.append((now, terminal))
        if not producer_terminal and queue is not None:
            while not queue.empty():
                queue.get_nowait()
            queue.put_nowait(terminal)
        logger.info('SSE stream request %s получил terminal event', request_id)
        return terminal

    def _remember_event(
        self,
        state: _RequestStreamState,
        event: SequencedStreamEvent,
        now: float,
        *,
        allow_roll: bool,
    ) -> None:
        if len(state.buffer) >= self._buffer_max_events:
            if not allow_roll:
                raise RuntimeError('Late-connect buffer overflow must be terminalized')
            _, dropped = state.buffer.popleft()
            state.replay_floor_event_id = max(
                state.replay_floor_event_id,
                dropped.event_id,
            )
        state.buffer.append((now, event))

    def _free_buffer_slot_if_needed(self, request_id: str, now: float) -> None:
        buffered_states = [
            (buffered_request_id, state)
            for buffered_request_id, state in self._states.items()
            if buffered_request_id != request_id
            and state.queue is None
            and state.buffer
            and state.terminal_event is None
        ]
        if len(buffered_states) < self._buffer_max_requests:
            return
        oldest_request_id, oldest_state = min(
            buffered_states,
            key=lambda item: item[1].last_activity,
        )
        if oldest_state.has_had_subscriber:
            self._discard_replay_buffer(oldest_state)
            logger.warning(
                '⚠️ Replay buffer запроса %s очищен по общему лимиту',
                oldest_request_id,
            )
        else:
            self._terminalize(
                oldest_request_id,
                oldest_state,
                detail=LATE_BUFFER_OVERFLOW_DETAIL,
                now=now,
                producer_terminal=False,
            )
            logger.warning(
                '⚠️ Late-connect buffer запроса %s закрыт по общему лимиту',
                oldest_request_id,
            )

    def _prune_expired_state(self, now: float) -> None:
        for request_id, state in list(self._states.items()):
            if (
                state.terminal_event is None
                and (state.queue is not None or state.has_had_subscriber)
            ):
                while (
                    state.buffer
                    and now - state.buffer[0][0] > self._buffer_seconds
                ):
                    _, dropped = state.buffer.popleft()
                    state.replay_floor_event_id = max(
                        state.replay_floor_event_id,
                        dropped.event_id,
                    )

            if (
                state.terminal_event is None
                and state.queue is None
                and not state.has_had_subscriber
                and state.buffer
                and now - state.buffer[0][0] > self._buffer_seconds
            ):
                self._terminalize(
                    request_id,
                    state,
                    detail=LATE_BUFFER_EXPIRED_DETAIL,
                    now=now,
                    producer_terminal=False,
                )
                logger.warning(
                    '⚠️ Просроченный buffer запроса %s очищен без подключения клиента',
                    request_id,
                )

            if state.terminal_event is None and now >= state.deadline_at:
                self._terminalize(
                    request_id,
                    state,
                    detail=REQUEST_DEADLINE_DETAIL,
                    now=now,
                    producer_terminal=False,
                )

            if state.queue is not None or state.terminal_at is None:
                continue
            # Локальный overflow/deadline не означает остановку producer.
            # Пока producer не прислал свой terminal, tombstone нельзя удалять:
            # иначе поздний publish создаст новый state и второй terminal.
            # Общий state cap сохраняет жёсткую границу памяти даже для зависших
            # producer; при исчерпании новые streams получают controlled 503.
            if (
                state.producer_finished
                and now
                > max(state.terminal_at, state.last_activity)
                + self._buffer_seconds
            ):
                del self._states[request_id]

    @staticmethod
    def _discard_replay_buffer(state: _RequestStreamState) -> None:
        while state.buffer:
            _, dropped = state.buffer.popleft()
            state.replay_floor_event_id = max(
                state.replay_floor_event_id,
                dropped.event_id,
            )

    async def _cleanup_loop(self) -> None:
        while True:
            await asyncio.sleep(self._cleanup_interval_seconds)
            self._prune_expired_state(self._clock())
