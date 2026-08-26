from contextvars import ContextVar, Token
from dataclasses import dataclass


@dataclass
class AgentRequestTraceData:
    """Безопасные агрегаты одного RabbitMQ-запроса для корневого span."""

    request_id: str | None = None
    """Идентификатор обрабатываемого запроса.

    Нужен не только телеметрии: мутирующий MCP-вызов передаёт его как
    idempotency key, чтобы повторный захват реплики после сбоя не отправил
    консультацию второй раз (VERA-014)."""

    route: str = 'unknown'
    route_reason: str | None = None
    search_required: bool = False
    search_unavailable: bool = False
    search_chunk_count: int = 0
    tool_call_count: int = 0
    request_retry_count: int = 0
    mcp_retry_count: int = 0
    response_chunk_count: int = 0
    response_char_count: int = 0
    streaming_started: bool = False
    mutating_tool_called: bool = False
    consultation_email_status: str | None = None
    consultation_email_error_code: str | None = None
    outcome: str = 'unknown'


_current_request_trace: ContextVar[AgentRequestTraceData | None] = ContextVar(
    'agent_request_trace', default=None
)


def set_request_trace(data: AgentRequestTraceData) -> Token[AgentRequestTraceData | None]:
    return _current_request_trace.set(data)


def get_request_trace() -> AgentRequestTraceData | None:
    return _current_request_trace.get()


def reset_request_trace(token: Token[AgentRequestTraceData | None]) -> None:
    _current_request_trace.reset(token)
