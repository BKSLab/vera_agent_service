from typing import Annotated, NotRequired, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """Состояние диалога агента Веры между узлами графа (LangGraph).

    См. AGENT_VERA_ARCHITECTURE.md, раздел "Состояние диалога", и
    AGENT_SERVICE_PLAN.md, Этап 1/раздел 0.1.
    """

    session_id: str
    user_id: str | None
    """None — незалогиненный пользователь; общие консультационные тулы
    доступны и без авторизации."""

    messages: Annotated[list[BaseMessage], add_messages]
    """Reducer `add_messages` обязателен: новое сообщение, переданное в
    `graph.ainvoke`, ДОПОЛНЯЕТ историю, восстановленную Redis-checkpointer'ом
    по `session_id` (Этап 5), а не затирает её. Это единственный механизм,
    которым в графе реально работает решение "убрать `history` из payload
    RabbitMQ" (AGENT_SERVICE_PLAN.md, раздел 0.1, раздел 3.1)."""

    pii_aliases: NotRequired[dict[str, list[str]]]
    """Сессионное соответствие обезличенных маркеров реальным значениям.
    В поле попадают только значения из пользовательских сообщений. Оно
    сохраняется Redis-checkpointer'ом, но не включается в сообщения,
    передаваемые LLM. Consumer восстанавливает mapping до редактирования новой
    реплики, чтобы один alias не получил разные значения между репликами."""

    retrieved_chunks: list[dict]
    """Чанки, полученные от `vera_rag_kb` (Этап 4). Пустой список — либо тул
    не вызывался, либо RAG честно вернул "нет ответа" (см. `search_unavailable`
    ниже для различия этого случая от технической недоступности поиска)."""

    tool_calls: list[str]
    """Имена инструментов, вызванных в рамках текущей реплики пользователя —
    для наблюдаемости и детектирования зацикливания (Этап 9)."""

    search_unavailable: bool
    """True, если `vera_rag_kb` не вызывался из-за недоступности MCP Tools
    Server (`McpUnavailableError`, Этап 4.2) — отличает "в базе знаний нет
    ответа" (`retrieved_chunks == []`, `search_unavailable == False`) от
    "поиск сейчас технически недоступен" (оба `[]`/`True`). Узел
    `generate_with_context` (Этап 4.3) формирует разные ответы пользователю
    для этих двух случаев — см. AGENT_SERVICE_PLAN.md, раздел 0.1. Каждый
    новый вызов графа должен явно передавать `False` — поле не участвует в
    накоплении между репликами (в отличие от `messages`)."""
