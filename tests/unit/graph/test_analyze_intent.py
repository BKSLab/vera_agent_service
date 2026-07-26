import json

import httpx
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

from app.graph.nodes.analyze_intent import create_analyze_intent_node
from app.observability.request_trace import (
    AgentRequestTraceData,
    reset_request_trace,
    set_request_trace,
)
from tests.unit.graph._mock_llm import chat_model_with_handler


@tool
async def vera_rag_kb(query: str) -> dict:
    """Поиск по базе знаний о правах людей с инвалидностью."""
    return {'chunks': []}


@tool
async def send_consultation_email(consultation_text: str, email: str) -> dict:
    """Отправка итоговой консультации в PDF на email."""
    return {'status': 'ok', 'email': email, 'document_name': 'consultation.pdf'}


def _state(text: str):
    return {
        'session_id': 's',
        'user_id': None,
        'messages': [HumanMessage(content=text)],
        'retrieved_chunks': [],
        'tool_calls': [],
        'search_unavailable': False,
    }


def _completion(message: dict, finish_reason: str) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            'id': 'x',
            'object': 'chat.completion',
            'created': 1,
            'model': 'test-model',
            'choices': [{'index': 0, 'message': message, 'finish_reason': finish_reason}],
            'usage': {'prompt_tokens': 1, 'completion_tokens': 1, 'total_tokens': 2},
        },
    )


def _tool_call_response(name: str, arguments: dict) -> httpx.Response:
    return _completion(
        {
            'role': 'assistant',
            'content': None,
            'tool_calls': [
                {
                    'id': 'call_1',
                    'type': 'function',
                    'function': {
                        'name': name,
                        'arguments': json.dumps(arguments),
                    },
                }
            ],
        },
        finish_reason='tool_calls',
    )


def _direct_response(content: str) -> httpx.Response:
    return _completion({'role': 'assistant', 'content': content}, finish_reason='stop')


async def test_returns_tool_call_message_when_tool_needed():
    chat_model = chat_model_with_handler(
        lambda request: _tool_call_response('vera_rag_kb', {'query': 'квота'}),
        streaming=False,
    )
    node = create_analyze_intent_node(
        chat_model,
        vera_rag_kb,
        send_consultation_email,
    )

    trace_data = AgentRequestTraceData()
    token = set_request_trace(trace_data)
    try:
        result = await node(_state('Какая квота на трудоустройство инвалидов?'))
    finally:
        reset_request_trace(token)

    assert 'messages' in result
    ai_message = result['messages'][0]
    assert ai_message.tool_calls
    assert ai_message.tool_calls[0]['name'] == 'vera_rag_kb'
    assert ai_message.tool_calls[0]['args'] == {'query': 'квота'}
    assert trace_data.route == 'knowledge_base'
    assert trace_data.search_required is True


async def test_returns_empty_update_when_tool_not_needed():
    """Ответ модели без tool_calls НЕ сохраняется в messages — реальный
    ответ пользователю формирует отдельный вызов generate_direct (раздел
    0.1)."""
    chat_model = chat_model_with_handler(lambda request: _direct_response('Привет! Чем могу помочь?'), streaming=False)
    node = create_analyze_intent_node(
        chat_model,
        vera_rag_kb,
        send_consultation_email,
    )

    trace_data = AgentRequestTraceData()
    token = set_request_trace(trace_data)
    try:
        result = await node(_state('привет'))
    finally:
        reset_request_trace(token)

    assert result == {}
    assert trace_data.route == 'direct'
    assert trace_data.search_required is False


async def test_routes_explicit_consultation_email_request_to_mutating_tool():
    arguments = {
        'consultation_text': 'Итоговая консультация без предварительного форматирования.',
        'email': 'user@example.com',
    }
    chat_model = chat_model_with_handler(
        lambda request: _tool_call_response(
            'send_consultation_email',
            arguments,
        ),
        streaming=False,
    )
    node = create_analyze_intent_node(
        chat_model,
        vera_rag_kb,
        send_consultation_email,
    )

    trace_data = AgentRequestTraceData()
    token = set_request_trace(trace_data)
    try:
        result = await node(_state('Отправь консультацию на user@example.com'))
    finally:
        reset_request_trace(token)

    ai_message = result['messages'][0]
    assert ai_message.tool_calls[0]['name'] == 'send_consultation_email'
    assert ai_message.tool_calls[0]['args'] == arguments
    assert trace_data.route == 'consultation_email'
    assert trace_data.search_required is False
