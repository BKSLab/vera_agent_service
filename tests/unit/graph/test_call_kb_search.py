import pytest

from app.core.settings import McpSettings
from app.graph.nodes.call_kb_search import SEARCH_UNAVAILABLE_TOOL_MESSAGE, create_call_kb_search_node
from app.observability.request_trace import (
    AgentRequestTraceData,
    reset_request_trace,
    set_request_trace,
)


class _FakeTool:
    name = 'vera_rag_kb'

    def __init__(self, result=None, error: Exception | None = None):
        self._result = result
        self._error = error
        self.received_arguments: dict | None = None

    async def ainvoke(self, arguments: dict):
        self.received_arguments = arguments
        if self._error is not None:
            raise self._error
        return self._result


def _state_with_tool_call(query: str = 'квота', audience: str = 'both'):
    return {
        'session_id': 's',
        'user_id': None,
        'messages': [
            _AIMessageStub(
                tool_calls=[
                    {'id': 'call_1', 'name': 'vera_rag_kb', 'args': {'query': query, 'audience': audience}}
                ]
            )
        ],
        'retrieved_chunks': [],
        'tool_calls': [],
        'search_unavailable': False,
    }


class _AIMessageStub:
    """Только то, что использует call_kb_search: tool_calls на последнем
    сообщении — не нужен полноценный langchain_core.messages.AIMessage."""

    def __init__(self, tool_calls):
        self.tool_calls = tool_calls


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    async def _instant_sleep(_seconds):
        return None

    monkeypatch.setattr('app.clients.mcp_client.asyncio.sleep', _instant_sleep)


async def _run_node(tool: _FakeTool, retries: int = 1):
    settings = McpSettings(mcp_call_retries=retries, mcp_call_timeout_seconds=1.0)
    node = create_call_kb_search_node(tool, settings)
    trace_data = AgentRequestTraceData(route='knowledge_base', search_required=True)
    token = set_request_trace(trace_data)
    try:
        result = await node(_state_with_tool_call())
    finally:
        reset_request_trace(token)
    return result, trace_data


async def test_successful_search_updates_state_with_chunks():
    chunks = [{'chunk_id': 'c1', 'text': 'квота 2 процента'}]
    tool = _FakeTool(result=[{'type': 'text', 'text': '{"chunks": [{"chunk_id": "c1", "text": "квота 2 процента"}]}'}])

    result, trace_data = await _run_node(tool)

    assert result['retrieved_chunks'] == chunks
    assert result['search_unavailable'] is False
    assert result['tool_calls'] == ['vera_rag_kb']
    tool_message = result['messages'][0]
    assert tool_message.tool_call_id == 'call_1'
    assert trace_data.tool_call_count == 1
    assert trace_data.search_chunk_count == 1
    assert trace_data.search_unavailable is False


async def test_empty_chunks_is_not_treated_as_unavailable():
    """RAG честно вернул "нет ответа" — search_unavailable остаётся False."""
    tool = _FakeTool(result=[{'type': 'text', 'text': '{"chunks": []}'}])

    result, trace_data = await _run_node(tool)

    assert result['retrieved_chunks'] == []
    assert result['search_unavailable'] is False
    assert trace_data.search_chunk_count == 0
    assert trace_data.outcome == 'unknown'


async def test_mcp_unavailable_sets_search_unavailable_flag():
    tool = _FakeTool(error=RuntimeError('RAG Service недоступен'))

    result, trace_data = await _run_node(tool, retries=1)

    assert result['retrieved_chunks'] == []
    assert result['search_unavailable'] is True
    assert result['tool_calls'] == ['vera_rag_kb']
    tool_message = result['messages'][0]
    assert tool_message.content == SEARCH_UNAVAILABLE_TOOL_MESSAGE
    assert tool_message.tool_call_id == 'call_1'
    assert trace_data.tool_call_count == 1
    assert trace_data.search_unavailable is True
    assert trace_data.outcome == 'degraded'


async def test_tool_receives_arguments_from_tool_call():
    tool = _FakeTool(result=[{'type': 'text', 'text': '{"chunks": []}'}])

    await _run_node(tool)

    assert tool.received_arguments == {'query': 'квота', 'audience': 'both'}
