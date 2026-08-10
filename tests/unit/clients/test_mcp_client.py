import asyncio

import pytest

from app.clients.mcp_client import (
    _parse_tool_result,
    build_consultation_email_tool_proxy,
    build_kb_search_tool_proxy,
    call_mutating_tool_once,
    call_tool_with_retry,
    get_mcp_client,
    get_tools_with_retry,
)
from app.core.settings import McpSettings
from app.exceptions.mcp import McpUnavailableError
from app.schemas.mcp_tool_results import ConsultationEmailToolResult, KbSearchToolResult


class _FakeTool:
    def __init__(
        self,
        results: list | None = None,
        exceptions: list[Exception | None] | None = None,
        name: str = 'vera_rag_kb',
    ):
        self.name = name
        self._results = results or []
        self._exceptions = exceptions or []
        self.call_count = 0
        self.received_arguments: list[dict] = []

    async def ainvoke(self, arguments: dict):
        index = self.call_count
        self.call_count += 1
        self.received_arguments.append(arguments)
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
    name = 'vera_rag_kb'

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


def test_mcp_transport_timeout_allows_long_consultation_call(monkeypatch):
    captured: dict = {}

    class _Client:
        def __init__(self, connections, **kwargs):
            captured['connections'] = connections
            captured['kwargs'] = kwargs

    monkeypatch.setattr('app.clients.mcp_client.MultiServerMCPClient', _Client)

    get_mcp_client(
        McpSettings(
            mcp_call_timeout_seconds=15.0,
            mcp_consultation_email_timeout_seconds=360.0,
        )
    )

    connection = captured['connections']['vera-tools']
    assert connection['timeout'] == 360.0
    assert connection['sse_read_timeout'] == 360.0


async def test_get_tools_with_retry_retries_then_succeeds():
    client = _FakeClient(tools=['tool-a'], exceptions=[RuntimeError('conn'), None])
    result = await get_tools_with_retry(client, retries=3, timeout_seconds=1.0)
    assert result == ['tool-a']
    assert client.call_count == 2


async def test_get_tools_with_retry_raises_after_exhausting_retries():
    client = _FakeClient(exceptions=[RuntimeError('a'), RuntimeError('b')])
    with pytest.raises(McpUnavailableError):
        await get_tools_with_retry(client, retries=2, timeout_seconds=1.0)


async def test_kb_search_proxy_uses_vera_rag_kb_public_name_and_resolves_remote_tool_once():
    remote_tool = _FakeTool(results=[{'chunks': []}, {'chunks': []}])
    client = _FakeClient(tools=[remote_tool])
    proxy = build_kb_search_tool_proxy(client)

    assert proxy.name == 'vera_rag_kb'

    await proxy.ainvoke({'query': 'квота'})
    await proxy.ainvoke({'query': 'льготы'})

    assert client.call_count == 1
    assert remote_tool.call_count == 2
    assert remote_tool.received_arguments == [{'query': 'квота'}, {'query': 'льготы'}]


async def test_consultation_email_proxy_resolves_remote_tool_once_and_preserves_arguments():
    remote_tool = _FakeTool(
        name='send_consultation_email',
        results=[
            {'status': 'ok', 'email': 'one@example.com'},
            {'status': 'ok', 'email': 'two@example.com'},
        ],
    )
    client = _FakeClient(tools=[_FakeTool(results=[]), remote_tool])
    proxy = build_consultation_email_tool_proxy(client)

    assert proxy.name == 'send_consultation_email'

    await proxy.ainvoke(
        {
            'consultation_text': 'Первая консультация',
            'email': 'one@example.com',
        }
    )
    await proxy.ainvoke(
        {
            'consultation_text': 'Вторая консультация',
            'email': 'two@example.com',
            'consultation_topic': 'Трудовые права',
        }
    )

    assert client.call_count == 1
    assert remote_tool.call_count == 2
    assert remote_tool.received_arguments == [
        {
            'consultation_text': 'Первая консультация',
            'consultation_topic': 'Консультация',
            'email': 'one@example.com',
        },
        {
            'consultation_text': 'Вторая консультация',
            'consultation_topic': 'Трудовые права',
            'email': 'two@example.com',
        },
    ]


async def test_consultation_email_proxy_replaces_blank_topic_with_default():
    remote_tool = _FakeTool(
        name='send_consultation_email',
        results=[{'status': 'ok', 'email': 'user@example.com'}],
    )
    client = _FakeClient(tools=[remote_tool])
    proxy = build_consultation_email_tool_proxy(client)

    await proxy.ainvoke(
        {
            'consultation_text': 'Консультация',
            'email': 'user@example.com',
            'consultation_topic': '   ',
        }
    )

    assert remote_tool.received_arguments == [
        {
            'consultation_text': 'Консультация',
            'consultation_topic': 'Консультация',
            'email': 'user@example.com',
        }
    ]


async def test_call_tool_with_retry_parses_text_content_block():
    tool = _FakeTool(results=[[{'type': 'text', 'text': '{"chunks": []}'}]])
    result = await call_tool_with_retry(
        tool, {'query': 'q'}, retries=3, timeout_seconds=1.0, result_schema=KbSearchToolResult
    )
    assert result == {'chunks': []}


async def test_call_tool_with_retry_retries_on_tool_execution_error_then_succeeds():
    tool = _FakeTool(
        results=[None, [{'type': 'text', 'text': '{"chunks": [{"chunk_id": "c1"}]}'}]],
        exceptions=[RuntimeError('MCP tool failed'), None],
    )
    result = await call_tool_with_retry(
        tool, {'query': 'q'}, retries=3, timeout_seconds=1.0, result_schema=KbSearchToolResult
    )
    assert result == {'chunks': [{'chunk_id': 'c1'}]}
    assert tool.call_count == 2


async def test_call_tool_with_retry_raises_after_exhausting_retries():
    tool = _FakeTool(results=[None, None], exceptions=[RuntimeError('a'), RuntimeError('b')])
    with pytest.raises(McpUnavailableError):
        await call_tool_with_retry(
            tool, {'query': 'q'}, retries=2, timeout_seconds=1.0, result_schema=KbSearchToolResult
        )


async def test_call_tool_with_retry_times_out():
    with pytest.raises(McpUnavailableError):
        await call_tool_with_retry(
            _HangingTool(), {'query': 'q'}, retries=1, timeout_seconds=0.05, result_schema=KbSearchToolResult
        )


async def test_call_tool_with_retry_rejects_multi_block_response():
    """Несколько content-блоков вместо одного — сломанный протокол, не
    повод молча разобрать первый (VERA-021)."""
    tool = _FakeTool(
        results=[
            [
                {'type': 'text', 'text': '{"chunks": []}'},
                {'type': 'text', 'text': '{"chunks": []}'},
            ]
        ]
    )
    with pytest.raises(McpUnavailableError):
        await call_tool_with_retry(
            tool, {'query': 'q'}, retries=1, timeout_seconds=1.0, result_schema=KbSearchToolResult
        )


async def test_call_tool_with_retry_rejects_result_failing_schema():
    """Результат не проходит форму KbSearchToolResult — chunks не список
    (VERA-021)."""
    tool = _FakeTool(results=[[{'type': 'text', 'text': '{"chunks": "not-a-list"}'}]])
    with pytest.raises(McpUnavailableError):
        await call_tool_with_retry(
            tool, {'query': 'q'}, retries=1, timeout_seconds=1.0, result_schema=KbSearchToolResult
        )


async def test_mutating_tool_is_called_once_and_result_is_parsed():
    tool = _FakeTool(
        name='send_consultation_email',
        results=[
            [
                {
                    'type': 'text',
                    'text': (
                        '{"status":"ok","email":"user@example.com",'
                        '"document_name":"consultation.pdf"}'
                    ),
                }
            ]
        ],
    )

    result = await call_mutating_tool_once(
        tool,
        {
            'consultation_text': 'Полный текст',
            'email': 'user@example.com',
        },
        timeout_seconds=1.0,
        result_schema=ConsultationEmailToolResult,
    )

    assert result['status'] == 'ok'
    assert tool.call_count == 1


async def test_mutating_tool_never_retries_after_execution_error():
    tool = _FakeTool(
        name='send_consultation_email',
        results=[None, {'status': 'ok'}],
        exceptions=[RuntimeError('connection lost'), None],
    )

    with pytest.raises(McpUnavailableError):
        await call_mutating_tool_once(
            tool,
            {
                'consultation_text': 'Полный текст',
                'email': 'user@example.com',
            },
            timeout_seconds=1.0,
            result_schema=ConsultationEmailToolResult,
        )

    assert tool.call_count == 1


async def test_mutating_tool_rejects_result_failing_schema():
    """Результат не проходит форму ConsultationEmailToolResult — `status`
    не строка (VERA-021)."""
    tool = _FakeTool(
        name='send_consultation_email',
        results=[[{'type': 'text', 'text': '{"status": [1, 2, 3]}'}]],
    )

    with pytest.raises(McpUnavailableError):
        await call_mutating_tool_once(
            tool,
            {'consultation_text': 'Полный текст', 'email': 'user@example.com'},
            timeout_seconds=1.0,
            result_schema=ConsultationEmailToolResult,
        )


def test_parse_tool_result_accepts_plain_dict():
    assert _parse_tool_result({'chunks': []}, KbSearchToolResult) == {'chunks': []}


def test_parse_tool_result_parses_text_content_block_list():
    raw = [{'type': 'text', 'text': '{"chunks": [{"chunk_id": "c1"}]}'}]
    assert _parse_tool_result(raw, KbSearchToolResult) == {'chunks': [{'chunk_id': 'c1'}]}


def test_parse_tool_result_raises_on_unexpected_format():
    with pytest.raises(McpUnavailableError):
        _parse_tool_result(42, KbSearchToolResult)


def test_parse_tool_result_rejects_multiple_content_blocks():
    raw = [
        {'type': 'text', 'text': '{"chunks": []}'},
        {'type': 'text', 'text': '{"chunks": []}'},
    ]
    with pytest.raises(McpUnavailableError):
        _parse_tool_result(raw, KbSearchToolResult)


def test_parse_tool_result_rejects_invalid_json_text_block():
    raw = [{'type': 'text', 'text': 'не json'}]
    with pytest.raises(McpUnavailableError):
        _parse_tool_result(raw, KbSearchToolResult)
