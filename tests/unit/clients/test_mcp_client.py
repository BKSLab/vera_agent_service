import asyncio

import pytest

from app.clients.mcp_client import _parse_tool_result, call_tool_with_retry, get_tools_with_retry
from app.exceptions.mcp import McpUnavailableError


class _FakeTool:
    name = 'kb_search'

    def __init__(self, results: list | None = None, exceptions: list[Exception | None] | None = None):
        self._results = results or []
        self._exceptions = exceptions or []
        self.call_count = 0

    async def ainvoke(self, arguments: dict):
        index = self.call_count
        self.call_count += 1
        if index < len(self._exceptions) and self._exceptions[index] is not None:
            raise self._exceptions[index]
        return self._results[index]


class _FakeClient:
    def __init__(self, tools: list | None = None, exceptions: list[Exception | None] | None = None):
        self._tools = tools
        self._exceptions = exceptions or []
        self.call_count = 0

    async def get_tools(self):
        index = self.call_count
        self.call_count += 1
        if index < len(self._exceptions) and self._exceptions[index] is not None:
            raise self._exceptions[index]
        return self._tools


class _HangingTool:
    name = 'kb_search'

    async def ainvoke(self, arguments: dict):
        await asyncio.sleep(10)


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    """Ретраи в тестах не должны реально ждать backoff между попытками."""

    async def _instant_sleep(_seconds):
        return None

    monkeypatch.setattr('app.clients.mcp_client.asyncio.sleep', _instant_sleep)


async def test_get_tools_with_retry_returns_tools_on_success():
    client = _FakeClient(tools=['tool-a'])
    result = await get_tools_with_retry(client, retries=3, timeout_seconds=1.0)
    assert result == ['tool-a']


async def test_get_tools_with_retry_retries_then_succeeds():
    client = _FakeClient(tools=['tool-a'], exceptions=[RuntimeError('conn'), None])
    result = await get_tools_with_retry(client, retries=3, timeout_seconds=1.0)
    assert result == ['tool-a']
    assert client.call_count == 2


async def test_get_tools_with_retry_raises_after_exhausting_retries():
    client = _FakeClient(exceptions=[RuntimeError('a'), RuntimeError('b')])
    with pytest.raises(McpUnavailableError):
        await get_tools_with_retry(client, retries=2, timeout_seconds=1.0)


async def test_call_tool_with_retry_parses_text_content_block():
    tool = _FakeTool(results=[[{'type': 'text', 'text': '{"chunks": []}'}]])
    result = await call_tool_with_retry(tool, {'query': 'q'}, retries=3, timeout_seconds=1.0)
    assert result == {'chunks': []}


async def test_call_tool_with_retry_retries_on_tool_execution_error_then_succeeds():
    tool = _FakeTool(
        results=[None, [{'type': 'text', 'text': '{"chunks": [{"chunk_id": "c1"}]}'}]],
        exceptions=[RuntimeError('MCP tool failed'), None],
    )
    result = await call_tool_with_retry(tool, {'query': 'q'}, retries=3, timeout_seconds=1.0)
    assert result == {'chunks': [{'chunk_id': 'c1'}]}
    assert tool.call_count == 2


async def test_call_tool_with_retry_raises_after_exhausting_retries():
    tool = _FakeTool(results=[None, None], exceptions=[RuntimeError('a'), RuntimeError('b')])
    with pytest.raises(McpUnavailableError):
        await call_tool_with_retry(tool, {'query': 'q'}, retries=2, timeout_seconds=1.0)


async def test_call_tool_with_retry_times_out():
    with pytest.raises(McpUnavailableError):
        await call_tool_with_retry(_HangingTool(), {'query': 'q'}, retries=1, timeout_seconds=0.05)


def test_parse_tool_result_accepts_plain_dict():
    assert _parse_tool_result({'chunks': []}) == {'chunks': []}


def test_parse_tool_result_parses_text_content_block_list():
    raw = [{'type': 'text', 'text': '{"chunks": [{"chunk_id": "c1"}]}'}]
    assert _parse_tool_result(raw) == {'chunks': [{'chunk_id': 'c1'}]}


def test_parse_tool_result_raises_on_unexpected_format():
    with pytest.raises(McpUnavailableError):
        _parse_tool_result(42)
