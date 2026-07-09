import json
from collections.abc import AsyncIterator

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app.streaming.session_bus import SessionBus


def create_sse_router(session_bus: SessionBus) -> APIRouter:
    """Создаёт роутер `GET /sse/{session_id}` (Этап 7.2, контракт — раздел
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

    @router.get('/sse/{session_id}')
    async def stream_session(session_id: str) -> StreamingResponse:
        return StreamingResponse(_event_stream(session_bus, session_id), media_type='text/event-stream')

    return router


async def _event_stream(session_bus: SessionBus, session_id: str) -> AsyncIterator[str]:
    queue = session_bus.subscribe(session_id)
    try:
        while True:
            event = await queue.get()
            yield f'data: {json.dumps(event, ensure_ascii=False)}\n\n'
            if event.get('type') in ('done', 'error'):
                break
    finally:
        # Срабатывает и при штатном завершении (done/error), и при обрыве
        # соединения клиентом (Starlette отменяет генератор — GeneratorExit).
        session_bus.unsubscribe(session_id, queue)
