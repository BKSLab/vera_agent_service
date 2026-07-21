from collections.abc import Callable, Coroutine
from typing import Any

from langchain_core.messages import AIMessage, SystemMessage
from langchain_openai import ChatOpenAI

from app.clients.llm import astream_tokens
from app.graph.prompts.context import (
    NO_ANSWER_INSTRUCTION,
    SEARCH_UNAVAILABLE_INSTRUCTION,
    format_chunks_instruction,
)
from app.graph.prompts.system import SYSTEM_PROMPT
from app.graph.state import AgentState


def create_generate_with_context_node(
    chat_model: ChatOpenAI,
) -> Callable[[AgentState], Coroutine[Any, Any, dict]]:
    """Создаёт узел `generate_with_context` (Этап 4.3) — стримингованная
    финальная генерация с чанками `vera_rag_kb` в контексте.

    Три ветки инструкции в зависимости от результата `call_kb_search`
    (раздел 0.1): есть релевантные чанки / база честно не нашла ответ /
    поиск технически недоступен — намеренно разные сообщения пользователю,
    не должны схлопываться в одинаковый текст.

    Стриминг токенов наружу (в SSE) не является ответственностью этого
    узла — узел лишь вызывает `model.astream()` через `astream_tokens`;
    внешний потребитель (RabbitMQ-consumer, Этап 6) слушает эти токены
    через `graph.astream_events(...)`, перехватывая события
    `on_chat_model_stream`, которые LangGraph автоматически генерирует по
    мере вызова модели внутри узла.
    """

    async def generate_with_context(state: AgentState) -> dict:
        if state.get('search_unavailable'):
            instruction = SEARCH_UNAVAILABLE_INSTRUCTION
        elif not state.get('retrieved_chunks'):
            instruction = NO_ANSWER_INSTRUCTION
        else:
            instruction = format_chunks_instruction(state['retrieved_chunks'])

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            *state['messages'],
            SystemMessage(content=instruction),
        ]

        full_text = ''
        async for token in astream_tokens(chat_model, messages):
            full_text += token

        return {'messages': [AIMessage(content=full_text)]}

    return generate_with_context
