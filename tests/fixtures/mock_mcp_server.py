"""Мок MCP Tools Server для тестов и локальной разработки Agent Service.

MCP Tools Server как отдельный сервис пока не существует (см.
AGENT_SERVICE_PLAN.md, раздел 0 — отдельный трек). Этот модуль поднимает
минимальный, но настоящий (не замоканный на уровне HTTP-транспорта) сервер
с единственным тулом `kb_search`, чтобы `app/clients/mcp_client.py` можно
было реализовать и протестировать против реального MCP-протокола (SSE)
уже сейчас — контракт зафиксирован в разделе 3.3 плана.
"""

import asyncio
import socket
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from mcp.server.fastmcp import FastMCP


def create_mock_mcp_app(
    chunks: list[dict] | None = None,
    *,
    fail_message: str | None = None,
    delay_seconds: float = 0.0,
) -> FastAPI:
    """Создаёт ASGI-приложение мок-сервера с тулом `kb_search`.

    Args:
        chunks: чанки, которые вернёт `kb_search` в поле `chunks` ответа
            (формат — как `POST /api/v1/search` в `vera_rag_service`,
            раздел 3.3 плана). Пустой список по умолчанию.
        fail_message: если задан, `kb_search` бросает `RuntimeError` с этим
            сообщением — имитация сбоя (RAG Service/MCP Tools Server
            недоступны).
        delay_seconds: задержка перед ответом — имитация медленного
            RAG Service, для тестов таймаута на стороне клиента.
    """
    app = FastAPI()
    mcp = FastMCP('vera-tools-mock')

    @mcp.tool()
    async def kb_search(query: str, audience: str = 'both') -> dict:
        """Поиск по базе знаний о правах людей с инвалидностью (мок)."""
        if delay_seconds:
            await asyncio.sleep(delay_seconds)
        if fail_message is not None:
            raise RuntimeError(fail_message)
        return {'chunks': chunks or []}

    app.mount('/mcp', mcp.sse_app())
    return app


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as temp_socket:
        temp_socket.bind(('127.0.0.1', 0))
        return temp_socket.getsockname()[1]


@asynccontextmanager
async def run_mock_mcp_server(app: FastAPI) -> AsyncIterator[str]:
    """Запускает мок-сервер на свободном локальном порту и отдаёт SSE-URL.

    Используется в интеграционных тестах `app/clients/mcp_client.py`
    (Этап 3.5) — соединение идёт по настоящему MCP/SSE-протоколу поверх
    HTTP, не через `httpx.MockTransport`.
    """
    port = _free_port()
    config = uvicorn.Config(app, host='127.0.0.1', port=port, log_level='warning')
    server = uvicorn.Server(config)
    server_task = asyncio.create_task(server.serve())
    try:
        while not server.started:
            await asyncio.sleep(0.02)
        yield f'http://127.0.0.1:{port}/mcp/sse'
    finally:
        server.should_exit = True
        await server_task
