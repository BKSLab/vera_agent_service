import json

from langchain_core.messages import HumanMessage, ToolMessage

from app.core.settings import GraphContextSettings
from app.graph.nodes.generate_direct import create_generate_direct_node
from app.graph.policy import CONSULTATION_EMAIL_GUARD_NOTICE
from tests.unit.graph._mock_llm import chat_model_with_handler, stream_response


def _state(guard_notice: str | None = None, messages=None):
    return {
        'session_id': 's',
        'user_id': None,
        'messages': messages or [HumanMessage(content='Привет!')],
        'retrieved_chunks': [],
        'tool_calls': [],
        'search_unavailable': False,
        'consultation_email_guard_notice': guard_notice,
    }


async def test_generate_direct_returns_accumulated_streamed_answer():
    chat_model = chat_model_with_handler(lambda request: stream_response(['Здравствуйте', '! Чем могу помочь?']))
    node = create_generate_direct_node(chat_model, GraphContextSettings())

    result = await node(_state())

    assert result['messages'][0].content == 'Здравствуйте! Чем могу помочь?'


async def test_generate_direct_does_not_call_model_when_email_send_was_blocked():
    """Отклонённый mutating tool-call должен завершаться кодовым ответом.

    Prompt не является защитой от побочного эффекта: модель может проигнорировать
    guard и вернуть `call:default_api:...` обычным текстом.
    """
    captured: dict = {}

    def handler(request):
        payload = json.loads(request.content)
        captured['system_messages'] = [
            message['content'] for message in payload['messages'] if message['role'] == 'system'
        ]
        return stream_response(['Подтвердите, пожалуйста, адрес ещё раз.'])

    chat_model = chat_model_with_handler(handler)
    node = create_generate_direct_node(chat_model, GraphContextSettings())

    result = await node(_state(guard_notice=CONSULTATION_EMAIL_GUARD_NOTICE))

    assert 'messages' not in captured
    assert result['messages'][0].content == (
        'Подтвердите, пожалуйста, адрес электронной почты и отправку консультации?'
    )


async def test_generate_direct_formats_email_result_without_llm():
    tool_message = ToolMessage(
        content=json.dumps(
            {
                'status': 'ok',
                'email': 'user@example.com',
                'document_name': 'consultation.pdf',
            }
        ),
        tool_call_id='call_1',
    )
    chat_model = chat_model_with_handler(lambda request: stream_response(['это не должно появиться']))
    node = create_generate_direct_node(chat_model, GraphContextSettings())

    result = await node(_state(messages=[HumanMessage(content='Отправь'), tool_message]))

    assert result['messages'][0].content == (
        'Документ отправлен на user@example.com. Название документа: consultation.pdf.'
    )


async def test_generate_direct_replaces_pseudo_tool_call_text_with_safe_response():
    chat_model = chat_model_with_handler(
        lambda request: stream_response(
            ['call:default_api:send_consultation_email{"email":"user@example.com"}']
        )
    )
    node = create_generate_direct_node(chat_model, GraphContextSettings())

    result = await node(_state())

    assert 'call:default_api:' not in result['messages'][0].content
    assert 'безопасный ответ' in result['messages'][0].content


async def test_generate_direct_does_not_add_guard_notice_when_not_blocked():
    captured: dict = {}

    def handler(request):
        payload = json.loads(request.content)
        captured['system_messages'] = [
            message['content'] for message in payload['messages'] if message['role'] == 'system'
        ]
        return stream_response(['Привет!'])

    chat_model = chat_model_with_handler(handler)
    node = create_generate_direct_node(chat_model, GraphContextSettings())

    await node(_state())

    assert CONSULTATION_EMAIL_GUARD_NOTICE not in captured['system_messages']
