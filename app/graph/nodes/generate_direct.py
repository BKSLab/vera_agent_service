import json
import logging
from collections.abc import Callable, Coroutine
from typing import Any

from langchain_core.messages import AIMessage, SystemMessage, ToolMessage

from app.clients.polza_final_response import FinalResponseGenerator
from app.core.settings import GraphContextSettings
from app.graph.context_budget import build_bounded_messages
from app.graph.output_guard import validate_plain_final_answer
from app.graph.policy import UNSAFE_TOOL_CALL_RESPONSE
from app.graph.prompts.context import NO_SEARCH_PERFORMED_INSTRUCTION
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


def _checked_ai_message(answer: str) -> AIMessage:
    """Не допускает небезопасный текст в state даже для кодовой email-ветки."""
    decision = validate_plain_final_answer(answer)
    if not decision.accepted or decision.answer is None:
        logger.error(
            '❌ Небезопасный ответ отклонён на границе generate_direct '
            '(reason=%s)',
            decision.reason,
        )
        answer = UNSAFE_TOOL_CALL_RESPONSE
    return AIMessage(content=answer)


def create_generate_direct_node(
    final_response_generator: FinalResponseGenerator,
    context_settings: GraphContextSettings,
) -> Callable[[AgentState], Coroutine[Any, Any, dict]]:
    """Создаёт узел `generate_direct` (Этап 4.4) — проверенный прямой ответ.

    Модели передаётся ограниченное по бюджету представление истории
    (`context_settings`, VERA-020), не вся `state['messages']`. Финальный
    ответ полностью буферизуется и проходит output guard до создания
    ``AIMessage``; сырой model stream из узла наружу не публикуется."""

    async def generate_direct(state: AgentState) -> dict:
        last_message = state['messages'][-1]
        if isinstance(last_message, ToolMessage):
            return {
                'messages': [
                    _checked_ai_message(
                        _format_consultation_email_result(last_message)
                    )
                ],
            }

        bounded_history = build_bounded_messages(
            state['messages'],
            max_turns=context_settings.context_max_turns,
            older_turns_summary_max_chars=context_settings.context_older_turns_summary_max_chars,
        )
        messages = [
            SystemMessage(content=FINAL_RESPONSE_SYSTEM_PROMPT),
            *bounded_history,
            SystemMessage(content=NO_SEARCH_PERFORMED_INSTRUCTION),
        ]

        answer = await final_response_generator.generate_final_answer(
            messages,
            node_name='generate_direct',
        )
        return {'messages': [_checked_ai_message(answer)]}

    return generate_direct
