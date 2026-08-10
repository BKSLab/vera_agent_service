import asyncio
import json
from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import APIRouter, Query, Response, status
from fastapi.responses import StreamingResponse

from app.exceptions.streaming import (
    InvalidStreamTicketError,
    SessionAlreadySubscribedError,
)
from app.streaming.session_bus import SessionBus
from app.streaming.ticket import StreamTicketVerifier


def create_sse_router(
    session_bus: SessionBus,
    ticket_verifier: StreamTicketVerifier,
) -> APIRouter:
    """Создаёт роутер `GET /sse/{request_id}` (Этап 7.2, контракт — раздел
    3.2 плана) поверх конкретного `SessionBus` — фабрика, а не глобальный
    объект, чтобы роутер оставался тестируемым на изолированном
    `SessionBus` (не на общем состоянии приложения).

    Формат событий:
    ```
    data: {"type": "token", "content": "..."}
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
    ) -> Response:
        try:
            ticket_verifier.verify(ticket, request_id=request_id)
        except InvalidStreamTicketError:
            return Response(status_code=status.HTTP_401_UNAUTHORIZED)

        try:
            queue = session_bus.subscribe(request_id)
        except SessionAlreadySubscribedError:
            return Response(status_code=status.HTTP_409_CONFLICT)

        return StreamingResponse(
            _event_stream(session_bus, request_id, queue),
            media_type='text/event-stream',
        )

    return router


async def _event_stream(
    session_bus: SessionBus,
    request_id: str,
    queue: asyncio.Queue,
) -> AsyncIterator[str]:
    try:
        while True:
            event = await queue.get()
            yield f'data: {json.dumps(event, ensure_ascii=False)}\n\n'
            if event.get('type') in ('done', 'error'):
                break
    finally:
        # Срабатывает и при штатном завершении (done/error), и при обрыве
        # соединения клиентом (Starlette отменяет генератор — GeneratorExit).
        session_bus.unsubscribe(request_id, queue)
