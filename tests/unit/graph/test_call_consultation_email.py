import json

from langchain_core.messages import AIMessage, HumanMessage

from app.core.settings import McpSettings
from app.graph.nodes.call_consultation_email import (
    UNRESOLVED_EMAIL_RESULT,
    create_call_consultation_email_node,
)
from app.observability.request_trace import (
    AgentRequestTraceData,
    reset_request_trace,
    set_request_trace,
)
from app.privacy.pii import pii_redaction_scope, redact_pii_text


class _FakeConsultationEmailTool:
    name = 'send_consultation_email'

    def __init__(self, result: dict | None = None, error: Exception | None = None):
        self._result = result
        self._error = error
        self.call_count = 0
        self.arguments: list[dict] = []

    async def ainvoke(self, arguments: dict):
        self.call_count += 1
        self.arguments.append(arguments)
        if self._error is not None:
            raise self._error
        return self._result


def _state(
    consultation_text: str = 'Полный текст консультации',
    email: str = 'user@example.com',
) -> dict:
    return {
        'session_id': 's',
        'user_id': None,
        'messages': [
            HumanMessage(content='Отправь консультацию'),
            AIMessage(
                content='',
                tool_calls=[
                    {
                        'id': 'call_1',
                        'name': 'send_consultation_email',
                        'args': {
                            'consultation_text': consultation_text,
                            'email': email,
                        },
                    }
                ],
            ),
        ],
        'retrieved_chunks': [],
        'tool_calls': [],
        'search_unavailable': False,
    }


async def _invoke_with_trace(tool: _FakeConsultationEmailTool):
    node = create_call_consultation_email_node(
        tool,
        McpSettings(mcp_consultation_email_timeout_seconds=1.0),
    )
    trace_data = AgentRequestTraceData()
    token = set_request_trace(trace_data)
    try:
        result = await node(_state())
    finally:
        reset_request_trace(token)
    return result, trace_data


async def test_success_result_is_forwarded_to_final_generation():
    tool_result = {
        'status': 'ok',
        'email': 'user@example.com',
        'document_name': 'consultation-2026-07-26.pdf',
    }
    tool = _FakeConsultationEmailTool(result=tool_result)

    result, trace_data = await _invoke_with_trace(tool)

    assert tool.call_count == 1
    assert tool.arguments == [
        {
            'consultation_text': 'Полный текст консультации',
            'email': 'user@example.com',
        }
    ]
    assert json.loads(result['messages'][0].content) == tool_result
    assert result['tool_calls'] == ['send_consultation_email']
    assert trace_data.mutating_tool_called is True
    assert trace_data.tool_call_count == 1
    assert trace_data.consultation_email_status == 'ok'
    assert trace_data.consultation_email_error_code is None


async def test_business_error_is_preserved_without_agent_retry():
    tool_result = {
        'status': 'error',
        'code': 'invalid_email',
        'message': 'Адрес электронной почты указан некорректно.',
    }
    tool = _FakeConsultationEmailTool(result=tool_result)

    result, trace_data = await _invoke_with_trace(tool)

    assert tool.call_count == 1
    assert json.loads(result['messages'][0].content) == tool_result
    assert trace_data.consultation_email_status == 'error'
    assert trace_data.consultation_email_error_code == 'invalid_email'


async def test_transport_error_becomes_unconfirmed_result_without_retry():
    tool = _FakeConsultationEmailTool(error=RuntimeError('connection lost'))

    result, trace_data = await _invoke_with_trace(tool)

    payload = json.loads(result['messages'][0].content)
    assert tool.call_count == 1
    assert payload['status'] == 'error'
    assert payload['code'] == 'delivery_unconfirmed'
    assert 'Проверьте почту' in payload['message']
    assert trace_data.mutating_tool_called is True
    assert trace_data.consultation_email_error_code == 'delivery_unconfirmed'


async def test_unexpected_result_is_not_reported_as_success():
    tool = _FakeConsultationEmailTool(result={'delivered': True})

    result, trace_data = await _invoke_with_trace(tool)

    payload = json.loads(result['messages'][0].content)
    assert payload['status'] == 'error'
    assert payload['code'] == 'unexpected_tool_result'
    assert trace_data.consultation_email_status == 'error'


async def test_email_alias_is_restored_only_immediately_before_tool_call():
    email = 'private.user@example.com'
    tool_result = {'status': 'ok', 'email': email}
    tool = _FakeConsultationEmailTool(result=tool_result)
    node = create_call_consultation_email_node(
        tool,
        McpSettings(mcp_consultation_email_timeout_seconds=1.0),
    )

    with pii_redaction_scope():
        redacted = redact_pii_text(f'Отправь консультацию на {email}')
        assert email not in redacted
        result = await node(_state(email='[EMAIL_1]'))

    assert tool.arguments == [
        {
            'consultation_text': 'Полный текст консультации',
            'email': email,
        }
    ]
    assert json.loads(result['messages'][0].content) == tool_result


async def test_unknown_email_alias_is_blocked_without_mcp_call():
    tool = _FakeConsultationEmailTool(result={'status': 'ok'})
    node = create_call_consultation_email_node(
        tool,
        McpSettings(mcp_consultation_email_timeout_seconds=1.0),
    )
    trace_data = AgentRequestTraceData()
    trace_token = set_request_trace(trace_data)
    try:
        with pii_redaction_scope():
            result = await node(_state(email='[EMAIL_99]'))
    finally:
        reset_request_trace(trace_token)

    assert tool.call_count == 0
    assert json.loads(result['messages'][0].content) == UNRESOLVED_EMAIL_RESULT
    assert trace_data.mutating_tool_called is False
    assert trace_data.tool_call_count == 0
    assert trace_data.consultation_email_status == 'error'
    assert trace_data.consultation_email_error_code == 'invalid_email'
