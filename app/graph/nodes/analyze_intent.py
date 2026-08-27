from collections.abc import Callable, Coroutine
from typing import Any

from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI

from app.clients.llm import ainvoke_with_retry
from app.clients.mcp_client import (
    SEND_CONSULTATION_EMAIL_TOOL_NAME,
    VERA_RAG_KB_TOOL_NAME,
)
from app.core.settings import GraphContextSettings
from app.graph.context_budget import build_bounded_messages
from app.graph.policy import (
    contains_legal_reference,
    find_last_human_message,
    is_probably_factual_or_legal_question,
    is_reference_only,
    is_simplify_answer_request,
)
from app.graph.prompts.system import SYSTEM_PROMPT
from app.graph.state import AgentState
from app.observability.request_trace import get_request_trace

FORCED_KB_SEARCH_TOOL_CALL_ID = 'forced-kb-search'
"""ID синтетического `tool_call`, которым код принудительно направляет
фактический/правовой вопрос в поиск по базе знаний, даже если модель решила
ответить напрямую (VERA-021)."""

REFERENCE_ONLY_KB_SEARCH_TOOL_CALL_ID = 'reference-only-kb-search'
"""ID детерминированного `tool_call` для реплики, целиком состоящей из
реквизитов нормы. Такая реплика не требует intent-LLM."""


def create_analyze_intent_node(
    chat_model: ChatOpenAI,
    kb_search_tool: BaseTool,
    consultation_email_tool: BaseTool,
    context_settings: GraphContextSettings,
) -> Callable[[AgentState], Coroutine[Any, Any, dict]]:
    """Создаёт узел `analyze_intent` (Этап 4.1).

    Короткий нестримингованный вызов — выбирает прямой ответ, поиск через
    `vera_rag_kb` или отправку через `send_consultation_email` и формирует
    аргументы выбранной тулы.

    Если тул не нужен, ответ модели **не сохраняется** в `messages` —
    реальный текст ответа пользователю формирует отдельный стримингованный
    вызов `generate_direct` (Этап 4.4), не этот узел.

    Отказ модели от поиска по базе знаний на фактическом/правовом вопросе
    дополнительно проверяется кодом (VERA-021).

    Два консервативных случая пропускают intent-LLM детерминированно: реплика,
    целиком состоящая из реквизитов нормы, сразу направляется в RAG; точный
    текст кнопки «Объяснить проще» после финального ответа сразу направляется
    в `generate_direct`.

    Модели передаётся не вся `state['messages']`, а ограниченное по
    бюджету представление (`context_settings`, VERA-020) — полная история
    остаётся в Redis-checkpoint'е и в PostgreSQL без изменений.
    """
    model_with_tools = chat_model.bind_tools(
        [kb_search_tool, consultation_email_tool]
    )

    async def analyze_intent(state: AgentState) -> dict:
        trace_data = get_request_trace()
        last_human = find_last_human_message(state['messages'])
        if (
            last_human is not None
            and isinstance(last_human.content, str)
            and is_reference_only(last_human.content)
        ):
            if trace_data is not None:
                trace_data.route = 'knowledge_base'
                trace_data.route_reason = 'reference_only'
                trace_data.search_required = True
            return {
                'messages': [
                    AIMessage(
                        content='',
                        tool_calls=[
                            {
                                'id': REFERENCE_ONLY_KB_SEARCH_TOOL_CALL_ID,
                                'name': VERA_RAG_KB_TOOL_NAME,
                                'args': {'query': last_human.content},
                            }
                        ],
                    )
                ]
            }

        if is_simplify_answer_request(state['messages']):
            if trace_data is not None:
                trace_data.route = 'direct'
                trace_data.route_reason = 'simplify_request'
            return {}

        bounded_history = build_bounded_messages(
            state['messages'],
            max_turns=context_settings.context_max_turns,
            older_turns_summary_max_chars=context_settings.context_older_turns_summary_max_chars,
        )
        messages = [SystemMessage(content=SYSTEM_PROMPT), *bounded_history]
        response = await ainvoke_with_retry(model_with_tools, messages)

        if response.tool_calls:
            tool_call = response.tool_calls[0]
            if trace_data is not None:
                trace_data.route_reason = 'model_tool_call'
                if tool_call['name'] == SEND_CONSULTATION_EMAIL_TOOL_NAME:
                    trace_data.route = 'consultation_email'
                else:
                    trace_data.route = 'knowledge_base'
                    trace_data.search_required = True
            return {'messages': [response]}

        if trace_data is not None:
            trace_data.route = 'direct'
            trace_data.route_reason = 'model_direct'

        has_legal_reference = (
            last_human is not None
            and isinstance(last_human.content, str)
            and contains_legal_reference(last_human.content)
        )
        if (
            last_human is not None
            and isinstance(last_human.content, str)
            and (
                has_legal_reference
                or is_probably_factual_or_legal_question(last_human.content)
            )
        ):
            # Модель решила ответить напрямую на вопрос, похожий на
            # фактический/правовой — код запрещает прямой ответ мимо базы
            # знаний (VERA-021) и принудительно направляет тот же вопрос в
            # vera_rag_kb; честный отказ, если поиск не найдёт ответа,
            # по-прежнему формирует generate_with_context.
            if trace_data is not None:
                trace_data.route = 'knowledge_base'
                trace_data.route_reason = (
                    'legal_reference'
                    if has_legal_reference
                    else 'factual_or_legal_keyword'
                )
                trace_data.search_required = True
            forced_message = AIMessage(
                content='',
                tool_calls=[
                    {
                        'id': FORCED_KB_SEARCH_TOOL_CALL_ID,
                        'name': VERA_RAG_KB_TOOL_NAME,
                        'args': {'query': last_human.content},
                    }
                ],
            )
            return {'messages': [forced_message]}

        return {}

    return analyze_intent
