from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """Состояние диалога агента Веры между узлами графа (LangGraph).

    См. AGENT_VERA_ARCHITECTURE.md, раздел "Состояние диалога", и
    AGENT_SERVICE_PLAN.md, Этап 1/раздел 0.1.
    """

    session_id: str
    user_id: str | None
    """None — незалогиненный пользователь (доступен только `kb_search`)."""

    messages: Annotated[list[BaseMessage], add_messages]
    """Reducer `add_messages` обязателен: новое сообщение, переданное в
    `graph.ainvoke`, ДОПОЛНЯЕТ историю, восстановленную Redis-checkpointer'ом
    по `session_id` (Этап 5), а не затирает её. Это единственный механизм,
    которым в графе реально работает решение "убрать `history` из payload
    RabbitMQ" (AGENT_SERVICE_PLAN.md, раздел 0.1, раздел 3.1)."""

    retrieved_chunks: list[dict]
    """Чанки, полученные от `kb_search` (Этап 4). Пустой список — либо тул
    не вызывался, либо RAG честно вернул "нет ответа"."""

    tool_calls: list[str]
    """Имена инструментов, вызванных в рамках текущей реплики пользователя —
    для наблюдаемости и детектирования зацикливания (Этап 9)."""
