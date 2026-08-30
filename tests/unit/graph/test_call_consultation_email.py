import json
import logging

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
from app.privacy.pii import (
    EMAIL_KIND,
    PASSPORT_KIND,
    PERSON_KIND,
    PHONE_KIND,
    SNILS_KIND,
    pii_redaction_scope,
    redact_pii_text,
)


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
    consultation_topic: str | None = None,
) -> dict:
    arguments = {
        'consultation_text': consultation_text,
        'email': email,
    }
    if consultation_topic is not None:
        arguments['consultation_topic'] = consultation_topic

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
                        'args': arguments,
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
        with pii_redaction_scope():
            redact_pii_text(
                'Отправьте на user@example.com',
                trusted=True,
            )
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
        redacted = redact_pii_text(
            f'Отправь консультацию на {email}',
            trusted=True,
        )
        assert email not in redacted
        result = await node(_state(email='[EMAIL_1]'))

    assert tool.arguments == [
        {
            'consultation_text': 'Полный текст консультации',
            'email': email,
        }
    ]
    assert json.loads(result['messages'][0].content) == tool_result


async def test_document_markers_are_neutralized_without_restoring_pii(caplog):
    email = 'user@example.com'
    person = 'Мария Петрова'
    phone = '8 495 111-22-33'
    snils = '123-456-789 00'
    passport = '63 12 123456'
    tool_result = {
        'status': 'ok',
        'email': email,
        'document_name': 'consultation.pdf',
    }
    tool = _FakeConsultationEmailTool(result=tool_result)
    node = create_call_consultation_email_node(
        tool,
        McpSettings(mcp_consultation_email_timeout_seconds=1.0),
    )
    caplog.set_level(logging.WARNING, logger='vera_agent_service')

    with pii_redaction_scope() as context:
        assert context.alias_for(EMAIL_KIND, email, trusted=True) == '[EMAIL_1]'
        assert context.alias_for(PERSON_KIND, person, trusted=True) == '[ФИО_1]'
        assert context.alias_for(PHONE_KIND, phone, trusted=True) == '[ТЕЛЕФОН_1]'
        assert context.alias_for(SNILS_KIND, snils, trusted=True) == '[СНИЛС_1]'
        assert context.alias_for(PASSPORT_KIND, passport, trusted=True) == '[ПАСПОРТ_1]'
        result = await node(
            _state(
                consultation_text=(
                    '[ФИО_1]; [EMAIL_1]; [ТЕЛЕФОН_1]; [СНИЛС_1]; '
                    '[ПАСПОРТ_1]; неизвестный алиас [ФИО_99].'
                ),
                consultation_topic='Документы [ФИО_2] и [ПАСПОРТ_99]',
                email='[EMAIL_1]',
            )
        )

    assert json.loads(result['messages'][0].content) == tool_result
    assert tool.call_count == 1
    arguments = tool.arguments[0]
    assert arguments['email'] == email
    assert arguments['consultation_text'] == (
        'указанное вами лицо; указанный адрес электронной почты; '
        'указанный номер телефона; указанные данные СНИЛС; '
        'указанные паспортные данные; неизвестный алиас указанное вами лицо.'
    )
    assert arguments['consultation_topic'] == (
        'Документы указанное вами лицо и указанные паспортные данные'
    )
    document = arguments['consultation_text'] + arguments['consultation_topic']
    for forbidden in (
        '[EMAIL_',
        '[ФИО_',
        '[ТЕЛЕФОН_',
        '[СНИЛС_',
        '[ПАСПОРТ_',
        email,
        person,
        phone,
        snils,
        passport,
    ):
        assert forbidden not in document

    messages = [
        record.getMessage()
        for record in caplog.records
        if 'Нейтрализованы маркеры ПДн' in record.getMessage()
    ]
    assert len(messages) == 1
    assert 'consultation_text=6' in messages[0]
    assert 'consultation_topic=2' in messages[0]
    assert 'заменено=8' in messages[0]
    assert 'ФИО=3' in messages[0]
    for forbidden in (email, person, phone, snils, passport, '[ФИО_1]'):
        assert forbidden not in messages[0]


async def test_document_without_markers_is_unchanged_and_not_logged(caplog):
    email = 'user@example.com'
    tool_result = {
        'status': 'ok',
        'email': email,
        'document_name': 'consultation.pdf',
    }
    tool = _FakeConsultationEmailTool(result=tool_result)
    node = create_call_consultation_email_node(
        tool,
        McpSettings(mcp_consultation_email_timeout_seconds=1.0),
    )
    caplog.set_level(logging.WARNING, logger='vera_agent_service')

    with pii_redaction_scope() as context:
        context.alias_for(EMAIL_KIND, email, trusted=True)
        await node(
            _state(
                consultation_text='По вашему обращению подготовлена консультация.',
                consultation_topic='Трудовые права',
                email='[EMAIL_1]',
            )
        )

    assert tool.arguments == [
        {
            'consultation_text': 'По вашему обращению подготовлена консультация.',
            'consultation_topic': 'Трудовые права',
            'email': email,
        }
    ]
    assert 'Нейтрализованы маркеры ПДн' not in caplog.text


async def test_untrusted_email_is_blocked_for_alias_and_raw_value():
    email = 'reference@example.com'
    tool = _FakeConsultationEmailTool(result={'status': 'ok'})
    node = create_call_consultation_email_node(
        tool,
        McpSettings(mcp_consultation_email_timeout_seconds=1.0),
    )

    with pii_redaction_scope():
        redacted = redact_pii_text(f'Справочный адрес: {email}')
        assert redacted == 'Справочный адрес: [EMAIL_1]'
        alias_result = await node(_state(email='[EMAIL_1]'))
        raw_result = await node(_state(email=email))

    assert tool.call_count == 0
    assert json.loads(alias_result['messages'][0].content) == UNRESOLVED_EMAIL_RESULT
    assert json.loads(raw_result['messages'][0].content) == UNRESOLVED_EMAIL_RESULT


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


async def test_missing_redaction_scope_blocks_email_without_mcp_call():
    tool = _FakeConsultationEmailTool(result={'status': 'ok'})
    node = create_call_consultation_email_node(
        tool,
        McpSettings(mcp_consultation_email_timeout_seconds=1.0),
    )

    result = await node(_state(email='user@example.com'))

    assert tool.call_count == 0
    assert json.loads(result['messages'][0].content) == UNRESOLVED_EMAIL_RESULT
