"""Мок MCP Tools Server для тестов и локальной разработки Agent Service.

Этот модуль поднимает минимальный, но настоящий (не замоканный на уровне
HTTP-транспорта) сервер с единственным тулом `vera_rag_kb`, чтобы
`app/clients/mcp_client.py` можно было протестировать против реального
MCP-протокола (streamable-http) без обращения к развёрнутому сервису.

Транспорт — streamable-http, не SSE (решение зафиксировано в
`vera_mcp_service/MCP_SERVICE_PLAN.md`, раздел 0.1/0.2, 2026-07-09) —
синхронизировано с реальным MCP Tools Server, который строится на том же
транспорте. `FastMCP.streamable_http_app()` — самостоятельное ASGI-приложение
с маршрутом уже на `/mcp` (не требует обёртки в FastAPI/дополнительного
`app.mount()`, в отличие от прежней SSE-версии этого файла).
"""

import asyncio
import socket
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import uvicorn
from mcp.server.fastmcp import FastMCP
from sse_starlette.sse import AppStatus
from starlette.applications import Starlette


def create_mock_mcp_app(
    chunks: list[dict] | None = None,
    *,
    fail_message: str | None = None,
    delay_seconds: float = 0.0,
) -> Starlette:
    """Создаёт ASGI-приложение мок-сервера с тулом `vera_rag_kb`.

    Args:
        chunks: чанки, которые вернёт `vera_rag_kb` в поле `chunks` ответа
            (формат — как `POST /api/v1/search` в `vera_rag_service`,
            раздел 3.3 плана). Пустой список по умолчанию.
        fail_message: если задан, `vera_rag_kb` бросает `RuntimeError` с этим
            сообщением — имитация сбоя (RAG Service/MCP Tools Server
            недоступны).
        delay_seconds: задержка перед ответом — имитация медленного
            RAG Service, для тестов таймаута на стороне клиента.
    """
    mcp = FastMCP('vera-tools-mock', stateless_http=True)

    @mcp.tool()
    async def vera_rag_kb(query: str, audience: str = 'both') -> dict:
        """Поиск по базе знаний о правах людей с инвалидностью (мок)."""
        if delay_seconds:
            await asyncio.sleep(delay_seconds)
        if fail_message is not None:
            raise RuntimeError(fail_message)
        return {'chunks': chunks or []}

    return mcp.streamable_http_app()


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as temp_socket:
        temp_socket.bind(('127.0.0.1', 0))
        return temp_socket.getsockname()[1]


@asynccontextmanager
async def run_mock_mcp_server(app: Starlette) -> AsyncIterator[str]:
    """Запускает мок-сервер на свободном локальном порту и отдаёт MCP-URL.

    Используется в интеграционных тестах `app/clients/mcp_client.py`
    (Этап 3.5) — соединение идёт по настоящему MCP/streamable-http протоколу
    поверх HTTP, не через `httpx.MockTransport`.

    **Находка (перепроверена для streamable-http, 2026-07-09):** та же
    проблема с `sse_starlette.sse.AppStatus.should_exit`, что и в прежней
    SSE-версии этого файла — только теперь актуальна не сама по себе для
    SSE-транспорта, а потому что streamable-http-транспорт `mcp` SDK тоже
    использует `sse_starlette.EventSourceResponse` внутри
    (`mcp/server/streamable_http.py`) для стрима ответов сервера. Флаг —
    процессный, не привязан к конкретному серверу: остановка одного
    мок-сервера взводит его и никогда не сбрасывает — следующий поднятый в
    этом же процессе сервер немедленно рвёт свои потоковые ответы
    ("peer closed connection without sending complete message body
    (incomplete chunked read)"), из-за чего клиент вешается на ожидании
    ответа на `initialize`. Подтверждено эмпирически: воспроизводится в
    прогоне из 2+ последовательных серверов в одном процессе, независимо
    пропадает при сбросе флага в `finally`.
    """
    port = _free_port()
    config = uvicorn.Config(app, host='127.0.0.1', port=port, log_level='warning')
    server = uvicorn.Server(config)
    server_task = asyncio.create_task(server.serve())
    try:
        while not server.started:
            await asyncio.sleep(0.02)
        yield f'http://127.0.0.1:{port}/mcp'
    finally:
        server.should_exit = True
        await server_task
        AppStatus.should_exit = False
