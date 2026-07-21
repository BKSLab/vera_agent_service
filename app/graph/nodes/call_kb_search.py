import json
from collections.abc import Callable, Coroutine
from typing import Any

from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool

from app.clients.mcp_client import call_tool_with_retry
from app.core.settings import McpSettings
from app.exceptions.mcp import McpUnavailableError
from app.graph.state import AgentState
from app.observability.request_trace import get_request_trace

SEARCH_UNAVAILABLE_TOOL_MESSAGE = 'Поиск по базе знаний временно недоступен.'


def create_call_kb_search_node(
    kb_search_tool: BaseTool, mcp_settings: McpSettings
) -> Callable[[AgentState], Coroutine[Any, Any, dict]]:
    """Создаёт узел `call_kb_search` (Этап 4.2) — вызывает тул через
    MCP-клиент (Этап 3). Ошибка MCP (`McpUnavailableError`) перехватывается
    здесь и не пробрасывается дальше — граф деградирует, не падает
    (AGENT_SERVICE_PLAN.md, раздел 0.1).

    Достижим только из `analyze_intent`, когда последнее сообщение — уже
    `AIMessage` с непустым `tool_calls` (маршрутизация — `app/graph/edges.py`),
    поэтому здесь этот инвариант не перепроверяется.
    """

    async def call_kb_search(state: AgentState) -> dict:
        tool_call = state['messages'][-1].tool_calls[0]
        trace_data = get_request_trace()
        if trace_data is not None:
            trace_data.tool_call_count += 1

        try:
            result = await call_tool_with_retry(
                kb_search_tool,
                tool_call['args'],
                retries=mcp_settings.mcp_call_retries,
                timeout_seconds=mcp_settings.mcp_call_timeout_seconds,
            )
        except McpUnavailableError:
            if trace_data is not None:
                trace_data.search_unavailable = True
                trace_data.search_chunk_count = 0
                trace_data.outcome = 'degraded'
            tool_message = ToolMessage(
                content=SEARCH_UNAVAILABLE_TOOL_MESSAGE, tool_call_id=tool_call['id']
            )
            return {
                'messages': [tool_message],
                'retrieved_chunks': [],
                'tool_calls': [tool_call['name']],
                'search_unavailable': True,
            }

        chunks = result.get('chunks', [])
        if trace_data is not None:
            trace_data.search_chunk_count = len(chunks)
        tool_message = ToolMessage(
            content=json.dumps(result, ensure_ascii=False), tool_call_id=tool_call['id']
        )
        return {
            'messages': [tool_message],
            'retrieved_chunks': chunks,
            'tool_calls': [tool_call['name']],
            'search_unavailable': False,
        }

    return call_kb_search
