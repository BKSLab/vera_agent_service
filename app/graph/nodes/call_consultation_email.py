import json
from collections.abc import Callable, Coroutine
from typing import Any

from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool

from app.clients.mcp_client import call_mutating_tool_once
from app.core.settings import McpSettings
from app.exceptions.mcp import McpUnavailableError
from app.graph.state import AgentState
from app.observability.request_trace import get_request_trace

UNCONFIRMED_DELIVERY_RESULT = {
    'status': 'error',
    'code': 'delivery_unconfirmed',
    'message': (
        'Не удалось подтвердить результат отправки консультации. '
        'Автоматический повтор не выполнялся: письмо могло быть принято '
        'почтовым сервером. Проверьте почту перед повторной отправкой.'
    ),
}

UNEXPECTED_RESULT = {
    'status': 'error',
    'code': 'unexpected_tool_result',
    'message': (
        'Сервис вернул неожиданный результат отправки консультации. '
        'Не сообщайте пользователю об успешной доставке.'
    ),
}


def create_call_consultation_email_node(
    consultation_email_tool: BaseTool,
    mcp_settings: McpSettings,
) -> Callable[[AgentState], Coroutine[Any, Any, dict]]:
    """Создаёт узел однократного вызова мутирующей email-тулы."""

    async def call_consultation_email(state: AgentState) -> dict:
        tool_call = state['messages'][-1].tool_calls[0]
        trace_data = get_request_trace()
        if trace_data is not None:
            trace_data.tool_call_count += 1
            # Ставится до сетевого вызова: после timeout результат отправки
            # неизвестен, поэтому consumer не должен перезапускать весь граф.
            trace_data.mutating_tool_called = True

        try:
            result = await call_mutating_tool_once(
                consultation_email_tool,
                tool_call['args'],
                timeout_seconds=(
                    mcp_settings.mcp_consultation_email_timeout_seconds
                ),
            )
        except McpUnavailableError:
            result = dict(UNCONFIRMED_DELIVERY_RESULT)

        if result.get('status') not in {'ok', 'error'}:
            result = dict(UNEXPECTED_RESULT)

        if trace_data is not None:
            trace_data.consultation_email_status = result['status']
            error_code = result.get('code')
            trace_data.consultation_email_error_code = (
                error_code if isinstance(error_code, str) else None
            )

        return {
            'messages': [
                ToolMessage(
                    content=json.dumps(result, ensure_ascii=False),
                    tool_call_id=tool_call['id'],
                )
            ],
            'tool_calls': [tool_call['name']],
        }

    return call_consultation_email
