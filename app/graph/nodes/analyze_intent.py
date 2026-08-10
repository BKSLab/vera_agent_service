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
    CONSULTATION_EMAIL_GUARD_NOTICE,
    consultation_email_send_is_confirmed,
    find_last_human_message,
    is_probably_factual_or_legal_question,
)
from app.graph.prompts.system import SYSTEM_PROMPT
from app.graph.state import AgentState
from app.observability.request_trace import get_request_trace

FORCED_KB_SEARCH_TOOL_CALL_ID = 'forced-kb-search'
"""ID синтетического `tool_call`, которым код принудительно направляет
фактический/правовой вопрос в поиск по базе знаний, даже если модель решила
ответить напрямую (VERA-021)."""


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

    Два решения модели дополнительно проверяются кодом (VERA-021), а не
    принимаются как есть: мутирующая отправка консультации и отказ от
    поиска по базе знаний на фактическом/правовом вопросе.

    Модели передаётся не вся `state['messages']`, а ограниченное по
    бюджету представление (`context_settings`, VERA-020) — полная история
    остаётся в Redis-checkpoint'е и в PostgreSQL без изменений.
    """
    model_with_tools = chat_model.bind_tools(
        [kb_search_tool, consultation_email_tool]
    )

    async def analyze_intent(state: AgentState) -> dict:
        bounded_history = build_bounded_messages(
            state['messages'],
            max_turns=context_settings.context_max_turns,
            older_turns_summary_max_chars=context_settings.context_older_turns_summary_max_chars,
        )
        messages = [SystemMessage(content=SYSTEM_PROMPT), *bounded_history]
        response = await ainvoke_with_retry(model_with_tools, messages)
        trace_data = get_request_trace()

        if response.tool_calls:
            tool_call = response.tool_calls[0]
            if (
                len(response.tool_calls) == 1
                and tool_call['name'] == SEND_CONSULTATION_EMAIL_TOOL_NAME
                and not consultation_email_send_is_confirmed(
                    state['messages'], tool_call['args'].get('email', '')
                )
            ):
                # Мутирующая отправка не выполняется по одному лишь
                # неподтверждённому намерению модели: граф возвращает
                # детерминированный запрос адреса/подтверждения, а не новый
                # LLM-вызов с инструкциями выбора tools.
                if trace_data is not None:
                    trace_data.route = 'direct'
                return {'consultation_email_guard_notice': CONSULTATION_EMAIL_GUARD_NOTICE}
            if trace_data is not None:
                if tool_call['name'] == SEND_CONSULTATION_EMAIL_TOOL_NAME:
                    trace_data.route = 'consultation_email'
                else:
                    trace_data.route = 'knowledge_base'
                    trace_data.search_required = True
            return {'messages': [response]}

        if trace_data is not None:
            trace_data.route = 'direct'

        last_human = find_last_human_message(state['messages'])
        if (
            last_human is not None
            and isinstance(last_human.content, str)
            and is_probably_factual_or_legal_question(last_human.content)
        ):
            # Модель решила ответить напрямую на вопрос, похожий на
            # фактический/правовой — код запрещает прямой ответ мимо базы
            # знаний (VERA-021) и принудительно направляет тот же вопрос в
            # vera_rag_kb; честный отказ, если поиск не найдёт ответа,
            # по-прежнему формирует generate_with_context.
            if trace_data is not None:
                trace_data.route = 'knowledge_base'
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
