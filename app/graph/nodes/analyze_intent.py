from collections.abc import Callable, Coroutine
from typing import Any

from langchain_core.messages import SystemMessage
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI

from app.clients.llm import ainvoke_with_retry
from app.graph.prompts.system import SYSTEM_PROMPT
from app.graph.state import AgentState
from app.observability.request_trace import get_request_trace


def create_analyze_intent_node(
    chat_model: ChatOpenAI, kb_search_tool: BaseTool
) -> Callable[[AgentState], Coroutine[Any, Any, dict]]:
    """Создаёт узел `analyze_intent` (Этап 4.1).

    Короткий нестримингованный вызов — решает, нужен ли `vera_rag_kb`, и с
    какими аргументами (раздел 0.1: фиксированно 2 вызова LLM на реплику).

    Если тул не нужен, ответ модели **не сохраняется** в `messages` —
    реальный текст ответа пользователю формирует отдельный стримингованный
    вызов `generate_direct` (Этап 4.4), не этот узел.
    """
    model_with_tools = chat_model.bind_tools([kb_search_tool])

    async def analyze_intent(state: AgentState) -> dict:
        messages = [SystemMessage(content=SYSTEM_PROMPT), *state['messages']]
        response = await ainvoke_with_retry(model_with_tools, messages)
        trace_data = get_request_trace()
        if response.tool_calls:
            if trace_data is not None:
                trace_data.route = 'knowledge_base'
                trace_data.search_required = True
            return {'messages': [response]}
        if trace_data is not None:
            trace_data.route = 'direct'
        return {}

    return analyze_intent
