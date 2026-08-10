import asyncio
import json
from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import APIRouter, Header, Query, Response, status
from fastapi.responses import StreamingResponse

from app.exceptions.streaming import (
    InvalidStreamTicketError,
    SessionAlreadySubscribedError,
)
from app.streaming.session_bus import (
    SequencedStreamEvent,
    SessionBus,
    SessionBusCapacityExceededError,
    SessionReplayCompleteError,
    SessionReplayUnavailableError,
)
from app.streaming.ticket import StreamTicketVerifier

HEARTBEAT_INTERVAL_SECONDS: float = 15.0


def create_sse_router(
    session_bus: SessionBus,
    ticket_verifier: StreamTicketVerifier,
    *,
    heartbeat_interval_seconds: float = HEARTBEAT_INTERVAL_SECONDS,
) -> APIRouter:
    """Создаёт роутер `GET /sse/{request_id}` (Этап 7.2, контракт — раздел
    3.2 плана) поверх конкретного `SessionBus` — фабрика, а не глобальный
    объект, чтобы роутер оставался тестируемым на изолированном
    `SessionBus` (не на общем состоянии приложения).

    Формат событий (каждое получает возрастающий ``id``):
    ```
    id: 1
    data: {"type": "token", "content": "..."}
    data: {"type": "heartbeat", "ts": 1700000000}
    data: {"type": "done"}
    data: {"type": "error", "detail": "..."}
    ```
    Поток завершается сам после `done`/`error` — это терминальные события
    (`app/messaging/consumer.py`, Этап 6, всегда шлёт ровно одно из них
    последним).
    """
    router = APIRouter()

    @router.get('/sse/{request_id}')
    async def stream_request(
        request_id: str,
        ticket: Annotated[str | None, Query()] = None,
        last_event_id: Annotated[
            str | None,
            Header(alias='Last-Event-ID'),
        ] = None,
    ) -> Response:
        try:
            ticket_verifier.verify(ticket, request_id=request_id)
        except InvalidStreamTicketError:
            return Response(status_code=status.HTTP_401_UNAUTHORIZED)

        try:
            queue = session_bus.subscribe(
                request_id,
                last_event_id=_parse_last_event_id(last_event_id),
            )
        except SessionAlreadySubscribedError:
            return Response(status_code=status.HTTP_409_CONFLICT)
        except SessionBusCapacityExceededError:
            return Response(status_code=status.HTTP_503_SERVICE_UNAVAILABLE)
        except SessionReplayUnavailableError:
            return Response(status_code=status.HTTP_410_GONE)
        except SessionReplayCompleteError:
            return Response(status_code=status.HTTP_204_NO_CONTENT)

        return StreamingResponse(
            _event_stream(
                session_bus,
                request_id,
                queue,
                heartbeat_interval_seconds=heartbeat_interval_seconds,
            ),
            media_type='text/event-stream',
        )

    return router


async def _event_stream(
    session_bus: SessionBus,
    request_id: str,
    queue: asyncio.Queue[SequencedStreamEvent],
    *,
    heartbeat_interval_seconds: float,
) -> AsyncIterator[str]:
    try:
        while True:
            remaining_seconds = session_bus.remaining_deadline_seconds(
                request_id,
            )
            if remaining_seconds <= 0:
                event = session_bus.terminalize_deadline(request_id)
            else:
                try:
                    event = await asyncio.wait_for(
                        queue.get(),
                        timeout=min(
                            heartbeat_interval_seconds,
                            remaining_seconds,
                        ),
                    )
                except TimeoutError:
                    try:
                        event = queue.get_nowait()
                    except asyncio.QueueEmpty:
                        if session_bus.remaining_deadline_seconds(request_id) <= 0:
                            event = session_bus.terminalize_deadline(
                                request_id,
                            )
                        else:
                            event = session_bus.create_heartbeat(request_id)

            yield _serialize_event(event)
            if event.payload.get('type') in ('done', 'error'):
                break
    finally:
        # Срабатывает и при штатном завершении (done/error), и при обрыве
        # соединения клиентом (Starlette отменяет генератор — GeneratorExit).
        session_bus.unsubscribe(request_id, queue)


def _serialize_event(event: SequencedStreamEvent) -> str:
    payload = json.dumps(event.payload, ensure_ascii=False)
    return f'id: {event.event_id}\ndata: {payload}\n\n'


def _parse_last_event_id(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        parsed = int(value)
    except ValueError:
        return -1
    return parsed
