import json

from langchain_core.messages import HumanMessage

from app.core.settings import GraphContextSettings
from app.graph.nodes.generate_direct import create_generate_direct_node
from app.graph.policy import CONSULTATION_EMAIL_GUARD_NOTICE
from tests.unit.graph._mock_llm import chat_model_with_handler, stream_response


def _state(guard_notice: str | None = None):
    return {
        'session_id': 's',
        'user_id': None,
        'messages': [HumanMessage(content='Привет!')],
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


async def test_generate_direct_adds_guard_notice_when_email_send_was_blocked():
    """VERA-021 (фикс регрессии): если `analyze_intent` отклонил tool_call
    и оставил guard notice в состоянии, `generate_direct` обязан передать
    модели инструкцию не изображать текстом вызов инструмента — иначе
    модель воспроизводит псевдовызов вида `call:default_api:...{...}`."""
    captured: dict = {}

    def handler(request):
        payload = json.loads(request.content)
        captured['system_messages'] = [
            message['content'] for message in payload['messages'] if message['role'] == 'system'
        ]
        return stream_response(['Подтвердите, пожалуйста, адрес ещё раз.'])

    chat_model = chat_model_with_handler(handler)
    node = create_generate_direct_node(chat_model, GraphContextSettings())

    await node(_state(guard_notice=CONSULTATION_EMAIL_GUARD_NOTICE))

    assert CONSULTATION_EMAIL_GUARD_NOTICE in captured['system_messages']


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
