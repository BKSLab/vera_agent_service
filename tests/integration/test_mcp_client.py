"""Интеграционные тесты `app/clients/mcp_client.py` против настоящего
локального MCP-сервера (tests/fixtures/mock_mcp_server.py) по протоколу
streamable-http, без обращения к развёрнутому `vera_mcp_service`."""

import pytest

from app.clients.mcp_client import call_tool_with_retry, get_mcp_client, get_tools_with_retry
from app.core.settings import McpSettings
from app.exceptions.mcp import McpUnavailableError
from tests.fixtures.mock_mcp_server import create_mock_mcp_app, run_mock_mcp_server

pytestmark = pytest.mark.integration


async def test_get_tools_and_call_kb_search_against_real_mock_server():
    chunks = [{'chunk_id': 'c1', 'text': 'квота 2 процента', 'score': 0.9}]
    app = create_mock_mcp_app(chunks=chunks)
    async with run_mock_mcp_server(app) as url:
        settings = McpSettings(mcp_server_url=url, mcp_call_timeout_seconds=5.0, mcp_call_retries=2)
        client = get_mcp_client(settings)

        tools = await get_tools_with_retry(
            client, retries=settings.mcp_call_retries, timeout_seconds=settings.mcp_call_timeout_seconds
        )
        assert [tool.name for tool in tools] == ['vera_rag_kb']

        result = await call_tool_with_retry(
            tools[0],
            {'query': 'квота', 'audience': 'both'},
            retries=settings.mcp_call_retries,
            timeout_seconds=settings.mcp_call_timeout_seconds,
        )
        assert result == {'chunks': chunks}


async def test_call_kb_search_returns_empty_chunks_when_rag_has_no_answer():
    """Пустой список — валидный ответ ("нет ответа на этот вопрос"), не
    ошибка (раздел 3.3 плана)."""
    app = create_mock_mcp_app(chunks=[])
    async with run_mock_mcp_server(app) as url:
        settings = McpSettings(mcp_server_url=url, mcp_call_timeout_seconds=5.0, mcp_call_retries=1)
        client = get_mcp_client(settings)
        tools = await get_tools_with_retry(client, retries=1, timeout_seconds=5.0)

        result = await call_tool_with_retry(
            tools[0], {'query': 'вопрос вне тематики БЗ'}, retries=1, timeout_seconds=5.0
        )
        assert result == {'chunks': []}


async def test_call_kb_search_raises_mcp_unavailable_when_tool_execution_fails():
    app = create_mock_mcp_app(fail_message='RAG Service недоступен (мок)')
    async with run_mock_mcp_server(app) as url:
        settings = McpSettings(mcp_server_url=url, mcp_call_timeout_seconds=5.0, mcp_call_retries=2)
        client = get_mcp_client(settings)
        tools = await get_tools_with_retry(client, retries=1, timeout_seconds=5.0)

        with pytest.raises(McpUnavailableError):
            await call_tool_with_retry(tools[0], {'query': 'q'}, retries=2, timeout_seconds=5.0)


async def test_call_kb_search_raises_mcp_unavailable_on_timeout():
    app = create_mock_mcp_app(chunks=[{'chunk_id': 'c1'}], delay_seconds=1.0)
    async with run_mock_mcp_server(app) as url:
        settings = McpSettings(mcp_server_url=url, mcp_call_timeout_seconds=5.0, mcp_call_retries=1)
        client = get_mcp_client(settings)
        tools = await get_tools_with_retry(client, retries=1, timeout_seconds=5.0)

        with pytest.raises(McpUnavailableError):
            await call_tool_with_retry(tools[0], {'query': 'q'}, retries=1, timeout_seconds=0.2)


async def test_get_tools_with_retry_raises_when_server_unreachable():
    settings = McpSettings(mcp_server_url='http://127.0.0.1:1/mcp', mcp_call_timeout_seconds=1.0)
    client = get_mcp_client(settings)

    with pytest.raises(McpUnavailableError):
        await get_tools_with_retry(client, retries=2, timeout_seconds=1.0)
