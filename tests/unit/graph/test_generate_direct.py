import json

from langchain_core.messages import HumanMessage, ToolMessage

from app.core.settings import GraphContextSettings
from app.graph.nodes.generate_direct import create_generate_direct_node
from app.graph.prompts.context import NO_SEARCH_PERFORMED_INSTRUCTION
from app.graph.prompts.system import FINAL_RESPONSE_SYSTEM_PROMPT
from tests.unit.graph._mock_llm import chat_model_with_handler, stream_response


def _state(*, messages=None):
    return {
        'session_id': 's',
        'user_id': None,
        'messages': messages or [HumanMessage(content='Привет!')],
        'retrieved_chunks': [],
        'tool_calls': [],
        'search_unavailable': False,
    }


def _system_message_contents(request) -> list[str]:
    payload = json.loads(request.content)
    return [message['content'] for message in payload['messages'] if message['role'] == 'system']


async def test_generate_direct_returns_accumulated_streamed_answer():
    chat_model = chat_model_with_handler(lambda request: stream_response(['Здравствуйте', '! Чем могу помочь?']))
    node = create_generate_direct_node(chat_model, GraphContextSettings())

    result = await node(_state())

    assert result['messages'][0].content == 'Здравствуйте! Чем могу помочь?'


async def test_generate_direct_appends_no_search_instruction_after_common_prompt():
    captured = {}

    def handler(request):
        captured['system_messages'] = _system_message_contents(request)
        return stream_response(['Уточните, пожалуйста, предмет вопроса.'])

    chat_model = chat_model_with_handler(handler)
    node = create_generate_direct_node(chat_model, GraphContextSettings())

    await node(_state())

    assert captured['system_messages'][0] == FINAL_RESPONSE_SYSTEM_PROMPT
    assert captured['system_messages'][-1] == NO_SEARCH_PERFORMED_INSTRUCTION
    assert NO_SEARCH_PERFORMED_INSTRUCTION not in FINAL_RESPONSE_SYSTEM_PROMPT


async def test_generate_direct_formats_email_result_without_calling_model():
    called = False

    def handler(request):
        nonlocal called
        called = True
        return stream_response(['этого ответа быть не должно'])

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
    chat_model = chat_model_with_handler(handler)
    node = create_generate_direct_node(chat_model, GraphContextSettings())

    result = await node(_state(messages=[HumanMessage(content='Отправь'), tool_message]))

    assert called is False
    assert result['messages'][0].content == (
        'Документ отправлен на user@example.com. Название документа: consultation.pdf.'
    )


async def test_generate_direct_replaces_pseudo_tool_call_with_safe_response():
    chat_model = chat_model_with_handler(
        lambda request: stream_response(
            ['call:default_api:', 'send_consultation_email{email=user@example.com}']
        )
    )
    node = create_generate_direct_node(chat_model, GraphContextSettings())

    result = await node(_state())

    assert 'call:default_api:' not in result['messages'][0].content
    assert 'безопасный ответ' in result['messages'][0].content
