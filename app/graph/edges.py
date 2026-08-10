from langchain_core.messages import AIMessage

from app.clients.mcp_client import (
    SEND_CONSULTATION_EMAIL_TOOL_NAME,
    VERA_RAG_KB_TOOL_NAME,
)
from app.graph.state import AgentState


def route_after_analyze_intent(state: AgentState) -> str:
    """Условное ребро после `analyze_intent` (Этап 4.5,
    `AGENT_VERA_ARCHITECTURE.md`, раздел "Граф агента Веры").

    Тул нужен ровно тогда, когда последнее сообщение — `AIMessage` с
    непустым `tool_calls` (это `analyze_intent` добавил его в `messages`,
    см. `app/graph/nodes/analyze_intent.py`). Если тул не нужен,
    `analyze_intent` не меняет `messages` — последним остаётся исходный
    `HumanMessage` пользователя, который не является `AIMessage`.

    Параллельные вызовы инструментов запрещены (VERA-021): больше одного
    `tool_calls` — ошибка маршрутизации модели, а не повод молча выбрать
    первый.
    """
    last_message = state['messages'][-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        if len(last_message.tool_calls) > 1:
            tool_names = [call['name'] for call in last_message.tool_calls]
            raise ValueError(
                f'Модель вернула параллельные вызовы инструментов, что запрещено: {tool_names}'
            )
        tool_name = last_message.tool_calls[0]['name']
        if tool_name == VERA_RAG_KB_TOOL_NAME:
            return 'call_kb_search'
        if tool_name == SEND_CONSULTATION_EMAIL_TOOL_NAME:
            return 'call_consultation_email'
        raise ValueError(f'Неподдерживаемый вызов инструмента: {tool_name}')
    return 'generate_direct'
