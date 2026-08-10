import json
import logging
from collections.abc import Callable, Coroutine
from typing import Any

from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI

from app.clients.llm import astream_tokens
from app.core.settings import GraphContextSettings
from app.exceptions.llm import EmptyLlmStreamError
from app.graph.context_budget import build_bounded_messages
from app.graph.policy import UNSAFE_TOOL_CALL_RESPONSE, contains_pseudo_tool_call
from app.graph.prompts.system import FINAL_RESPONSE_SYSTEM_PROMPT
from app.graph.state import AgentState
from app.schemas.mcp_tool_results import ConsultationEmailToolResult

logger = logging.getLogger('vera_agent_service')

def _format_consultation_email_result(tool_message: ToolMessage) -> str:
    """Формирует честный ответ по проверенному MCP-результату без LLM."""
    try:
        payload = json.loads(tool_message.content)
        result = ConsultationEmailToolResult.model_validate(payload)
    except (TypeError, ValueError):
        return 'Не удалось подтвердить результат отправки консультации. Проверьте почту.'

    if result.status == 'ok':
        response = 'Документ отправлен'
        if result.email:
            response += f' на {result.email}'
        response += '.'
        if result.document_name:
            response += f' Название документа: {result.document_name}.'
        return response

    if result.message:
        return f'Не удалось подтвердить отправку консультации. {result.message}'
    return 'Не удалось подтвердить результат отправки консультации. Проверьте почту.'


def create_generate_direct_node(
    chat_model: ChatOpenAI,
    context_settings: GraphContextSettings,
) -> Callable[[AgentState], Coroutine[Any, Any, dict]]:
    """Создаёт узел `generate_direct` (Этап 4.4) — стримингованный прямой
    ответ без вызова инструмента. См. docstring
    `create_generate_with_context_node` про механизм стриминга наружу.

    Модели передаётся ограниченное по бюджету представление истории
    (`context_settings`, VERA-020), не вся `state['messages']`."""

    async def generate_direct(state: AgentState) -> dict:
        guard_notice = state.get('consultation_email_guard_notice')
        if guard_notice:
            return {
                'messages': [AIMessage(content=guard_notice)],
                'consultation_email_guard_notice': None,
            }

        last_message = state['messages'][-1]
        if isinstance(last_message, ToolMessage):
            return {
                'messages': [AIMessage(content=_format_consultation_email_result(last_message))],
                'consultation_email_guard_notice': None,
            }

        bounded_history = build_bounded_messages(
            state['messages'],
            max_turns=context_settings.context_max_turns,
            older_turns_summary_max_chars=context_settings.context_older_turns_summary_max_chars,
        )
        messages = [SystemMessage(content=FINAL_RESPONSE_SYSTEM_PROMPT), *bounded_history]

        full_text = ''
        async for token in astream_tokens(chat_model, messages):
            full_text += token

        if not full_text:
            raise EmptyLlmStreamError

        if contains_pseudo_tool_call(full_text):
            logger.error('Заблокирован псевдовызов инструмента в тексте финального ответа')
            return {
                'messages': [AIMessage(content=UNSAFE_TOOL_CALL_RESPONSE)],
                'consultation_email_guard_notice': None,
            }

        return {
            'messages': [AIMessage(content=full_text)],
            'consultation_email_guard_notice': None,
        }

    return generate_direct
