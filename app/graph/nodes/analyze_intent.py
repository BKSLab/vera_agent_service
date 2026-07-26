from collections.abc import Callable, Coroutine
from typing import Any

from langchain_core.messages import SystemMessage
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI

from app.clients.llm import ainvoke_with_retry
from app.clients.mcp_client import SEND_CONSULTATION_EMAIL_TOOL_NAME
from app.graph.prompts.system import SYSTEM_PROMPT
from app.graph.state import AgentState
from app.observability.request_trace import get_request_trace


def create_analyze_intent_node(
    chat_model: ChatOpenAI,
    kb_search_tool: BaseTool,
    consultation_email_tool: BaseTool,
) -> Callable[[AgentState], Coroutine[Any, Any, dict]]:
    """Создаёт узел `analyze_intent` (Этап 4.1).

    Короткий нестримингованный вызов — выбирает прямой ответ, поиск через
    `vera_rag_kb` или отправку через `send_consultation_email` и формирует
    аргументы выбранной тулы.

    Если тул не нужен, ответ модели **не сохраняется** в `messages` —
    реальный текст ответа пользователю формирует отдельный стримингованный
    вызов `generate_direct` (Этап 4.4), не этот узел.
    """
    model_with_tools = chat_model.bind_tools(
        [kb_search_tool, consultation_email_tool]
    )

    async def analyze_intent(state: AgentState) -> dict:
        messages = [SystemMessage(content=SYSTEM_PROMPT), *state['messages']]
        response = await ainvoke_with_retry(model_with_tools, messages)
        trace_data = get_request_trace()
        if response.tool_calls:
            if trace_data is not None:
                tool_name = response.tool_calls[0]['name']
                if tool_name == SEND_CONSULTATION_EMAIL_TOOL_NAME:
                    trace_data.route = 'consultation_email'
                else:
                    trace_data.route = 'knowledge_base'
                    trace_data.search_required = True
            return {'messages': [response]}
        if trace_data is not None:
            trace_data.route = 'direct'
        return {}

    return analyze_intent
