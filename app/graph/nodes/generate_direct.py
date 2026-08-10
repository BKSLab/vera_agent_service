from collections.abc import Callable, Coroutine
from typing import Any

from langchain_core.messages import AIMessage, SystemMessage
from langchain_openai import ChatOpenAI

from app.clients.llm import astream_tokens
from app.core.settings import GraphContextSettings
from app.graph.context_budget import build_bounded_messages
from app.graph.prompts.system import SYSTEM_PROMPT
from app.graph.state import AgentState


def create_generate_direct_node(
    chat_model: ChatOpenAI,
    context_settings: GraphContextSettings,
) -> Callable[[AgentState], Coroutine[Any, Any, dict]]:
    """Создаёт узел `generate_direct` (Этап 4.4) — стримингованный прямой
    ответ без вызова инструмента. См. docstring
    `create_generate_with_context_node` про механизм стриминга наружу.

    Модели передаётся ограниченное по бюджету представление истории
    (`context_settings`, VERA-020), не вся `state['messages']`.

    Если код отклонил предложенный моделью mutating tool_call
    (`consultation_email_guard_notice`, VERA-021), добавляет одноразовую
    инструкцию для этого конкретного вызова — не в `state['messages']`,
    чтобы не засорять постоянную историю служебным сообщением."""

    async def generate_direct(state: AgentState) -> dict:
        bounded_history = build_bounded_messages(
            state['messages'],
            max_turns=context_settings.context_max_turns,
            older_turns_summary_max_chars=context_settings.context_older_turns_summary_max_chars,
        )
        messages = [SystemMessage(content=SYSTEM_PROMPT), *bounded_history]
        guard_notice = state.get('consultation_email_guard_notice')
        if guard_notice:
            messages.append(SystemMessage(content=guard_notice))

        full_text = ''
        async for token in astream_tokens(chat_model, messages):
            full_text += token

        return {'messages': [AIMessage(content=full_text)]}

    return generate_direct
