import logging
from collections.abc import Callable, Coroutine
from typing import Any

from langchain_core.messages import AIMessage, SystemMessage

from app.clients.polza_final_response import FinalResponseGenerator
from app.core.settings import GraphContextSettings
from app.graph.context_budget import build_bounded_messages
from app.graph.output_guard import validate_plain_final_answer
from app.graph.policy import UNSAFE_TOOL_CALL_RESPONSE
from app.graph.prompts.context import (
    NO_ANSWER_INSTRUCTION,
    SEARCH_UNAVAILABLE_INSTRUCTION,
    format_chunks_instruction,
)
from app.graph.prompts.system import FINAL_RESPONSE_SYSTEM_PROMPT
from app.graph.state import AgentState

logger = logging.getLogger('vera_agent_service')


def create_generate_with_context_node(
    final_response_generator: FinalResponseGenerator,
    context_settings: GraphContextSettings,
) -> Callable[[AgentState], Coroutine[Any, Any, dict]]:
    """Создаёт узел `generate_with_context` (Этап 4.3) — проверенная
    финальная генерация с чанками `vera_rag_kb` в контексте.

    Три ветки инструкции в зависимости от результата `call_kb_search`
    (раздел 0.1): есть релевантные чанки / база честно не нашла ответ /
    поиск технически недоступен — намеренно разные сообщения пользователю,
    не должны схлопываться в одинаковый текст.

    Финальный provider stream полностью буферизуется, reasoning отделяется,
    а ``AIMessage`` создаётся только после строгой проверки всего ответа.

    Модели передаётся ограниченное по бюджету представление истории
    (`context_settings`, VERA-020) — найденные чанки берутся из
    `state['retrieved_chunks']` независимо от бюджета, поэтому усечение
    старых реплик не влияет на ответ текущего turn'а.
    """

    async def generate_with_context(state: AgentState) -> dict:
        if state.get('search_unavailable'):
            instruction = SEARCH_UNAVAILABLE_INSTRUCTION
        elif not state.get('retrieved_chunks'):
            instruction = NO_ANSWER_INSTRUCTION
        else:
            instruction = format_chunks_instruction(state['retrieved_chunks'])

        bounded_history = build_bounded_messages(
            state['messages'],
            max_turns=context_settings.context_max_turns,
            older_turns_summary_max_chars=context_settings.context_older_turns_summary_max_chars,
        )
        messages = [
            SystemMessage(content=FINAL_RESPONSE_SYSTEM_PROMPT),
            *bounded_history,
            SystemMessage(content=instruction),
        ]

        answer = await final_response_generator.generate_final_answer(
            messages,
            node_name='generate_with_context',
        )
        decision = validate_plain_final_answer(answer)
        if not decision.accepted or decision.answer is None:
            logger.error(
                '❌ Небезопасный ответ отклонён на границе generate_with_context '
                '(reason=%s)',
                decision.reason,
            )
            answer = UNSAFE_TOOL_CALL_RESPONSE
        return {'messages': [AIMessage(content=answer)]}

    return generate_with_context
