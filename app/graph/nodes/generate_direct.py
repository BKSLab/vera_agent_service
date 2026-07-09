from collections.abc import Callable, Coroutine
from typing import Any

from langchain_core.messages import AIMessage, SystemMessage
from langchain_openai import ChatOpenAI

from app.clients.llm import astream_tokens
from app.graph.prompts.system import SYSTEM_PROMPT
from app.graph.state import AgentState


def create_generate_direct_node(
    chat_model: ChatOpenAI,
) -> Callable[[AgentState], Coroutine[Any, Any, dict]]:
    """Создаёт узел `generate_direct` (Этап 4.4) — стримингованный прямой
    ответ без вызова инструмента. См. docstring
    `create_generate_with_context_node` про механизм стриминга наружу."""

    async def generate_direct(state: AgentState) -> dict:
        messages = [SystemMessage(content=SYSTEM_PROMPT), *state['messages']]

        full_text = ''
        async for token in astream_tokens(chat_model, messages):
            full_text += token

        return {'messages': [AIMessage(content=full_text)]}

    return generate_direct
