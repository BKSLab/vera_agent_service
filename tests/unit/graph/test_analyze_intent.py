import json

import httpx
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.tools import tool

from app.core.settings import GraphContextSettings
from app.graph.nodes.analyze_intent import create_analyze_intent_node
from app.observability.request_trace import (
    AgentRequestTraceData,
    reset_request_trace,
    set_request_trace,
)
from tests.unit.graph._mock_llm import chat_model_with_handler

_CONTEXT_SETTINGS = GraphContextSettings()


@tool
async def vera_rag_kb(query: str) -> dict:
    """Поиск по базе знаний о правах людей с инвалидностью."""
    return {'chunks': []}


@tool
async def send_consultation_email(consultation_text: str, email: str) -> dict:
    """Отправка итоговой консультации в PDF на email."""
    return {'status': 'ok', 'email': email, 'document_name': 'consultation.pdf'}


def _state_with_messages(messages: list[BaseMessage]):
    return {
        'session_id': 's',
        'user_id': None,
        'messages': messages,
        'retrieved_chunks': [],
        'tool_calls': [],
        'search_unavailable': False,
    }


def _state(text: str):
    return _state_with_messages([HumanMessage(content=text)])


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
        _CONTEXT_SETTINGS,
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
        _CONTEXT_SETTINGS,
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


async def test_consultation_email_tool_call_is_allowed_for_explicit_request_with_email():
    """Явная просьба пользователя с адресом уже является подтверждением.

    Код проверяет именно пользовательский текст, а не одно только решение
    модели вызвать mutating tool (VERA-021).
    """
    arguments = {
        'consultation_text': 'Итоговая консультация без предварительного форматирования.',
        'email': 'user@example.com',
    }
    chat_model = chat_model_with_handler(
        lambda request: _tool_call_response('send_consultation_email', arguments),
        streaming=False,
    )
    node = create_analyze_intent_node(chat_model, vera_rag_kb, send_consultation_email, _CONTEXT_SETTINGS)

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


async def test_consultation_email_tool_call_is_guarded_when_email_never_typed_by_user():
    """Адрес, который пользователь не вводил, нельзя отправить по решению
    модели, даже если в истории уже был вопрос о подтверждении."""
    arguments = {'consultation_text': 'Итог консультации.', 'email': 'user@example.com'}
    chat_model = chat_model_with_handler(
        lambda request: _tool_call_response('send_consultation_email', arguments),
        streaming=False,
    )
    node = create_analyze_intent_node(chat_model, vera_rag_kb, send_consultation_email, _CONTEXT_SETTINGS)
    history = [
        HumanMessage(content='Отправь мне итог консультации на почту'),
        AIMessage(content='Подтвердите, пожалуйста, ваш email?'),
        HumanMessage(content='Да, отправляйте'),
    ]

    trace_data = AgentRequestTraceData()
    token = set_request_trace(trace_data)
    try:
        result = await node(_state_with_messages(history))
    finally:
        reset_request_trace(token)

    assert result == {
        'consultation_email_guard_notice': (
            'Подтвердите, пожалуйста, адрес электронной почты и отправку консультации?'
        )
    }
    assert trace_data.route == 'direct'


async def test_consultation_email_tool_call_is_allowed_when_address_was_given_in_earlier_turn():
    """Короткое «да» после переспроса не требует повторять ранее введённый
    адрес в той же реплике."""
    arguments = {'consultation_text': 'Итог консультации.', 'email': 'user@example.com'}
    chat_model = chat_model_with_handler(
        lambda request: _tool_call_response('send_consultation_email', arguments),
        streaming=False,
    )
    node = create_analyze_intent_node(chat_model, vera_rag_kb, send_consultation_email, _CONTEXT_SETTINGS)
    history = [
        HumanMessage(content='Мой email user@example.com, отправь консультацию'),
        AIMessage(content='Подтвердите отправку?'),
        HumanMessage(content='Да, отправляйте'),
    ]

    trace_data = AgentRequestTraceData()
    token = set_request_trace(trace_data)
    try:
        result = await node(_state_with_messages(history))
    finally:
        reset_request_trace(token)

    assert result['messages'][0].tool_calls[0]['name'] == 'send_consultation_email'
    assert trace_data.route == 'consultation_email'


async def test_consultation_email_tool_call_is_allowed_after_explicit_confirmation():
    """Оба кодовых факта выполнены: адрес в тексте текущего сообщения и
    запрос подтверждения в предыдущем ответе ассистента (VERA-021)."""
    arguments = {'consultation_text': 'Итог консультации.', 'email': 'user@example.com'}
    chat_model = chat_model_with_handler(
        lambda request: _tool_call_response('send_consultation_email', arguments),
        streaming=False,
    )
    node = create_analyze_intent_node(chat_model, vera_rag_kb, send_consultation_email, _CONTEXT_SETTINGS)
    history = [
        HumanMessage(content='Отправь мне итог консультации на почту'),
        AIMessage(content='Подтвердите, пожалуйста, ваш email?'),
        HumanMessage(content='Да, user@example.com'),
    ]

    trace_data = AgentRequestTraceData()
    token = set_request_trace(trace_data)
    try:
        result = await node(_state_with_messages(history))
    finally:
        reset_request_trace(token)

    ai_message = result['messages'][0]
    assert ai_message.tool_calls[0]['name'] == 'send_consultation_email'
    assert ai_message.tool_calls[0]['args'] == arguments
    assert trace_data.route == 'consultation_email'
    assert trace_data.search_required is False


async def test_factual_question_without_tool_call_is_forced_into_kb_search():
    """Модель ошибочно решила ответить напрямую на вопрос из предметной
    области базы знаний — код принудительно направляет его в поиск
    (VERA-021)."""
    chat_model = chat_model_with_handler(
        lambda request: _direct_response('Квота обычно небольшая.'),
        streaming=False,
    )
    node = create_analyze_intent_node(chat_model, vera_rag_kb, send_consultation_email, _CONTEXT_SETTINGS)

    trace_data = AgentRequestTraceData()
    token = set_request_trace(trace_data)
    try:
        result = await node(_state('Какая квота на трудоустройство инвалидов?'))
    finally:
        reset_request_trace(token)

    ai_message = result['messages'][0]
    assert ai_message.tool_calls[0]['name'] == 'vera_rag_kb'
    assert ai_message.tool_calls[0]['args'] == {
        'query': 'Какая квота на трудоустройство инвалидов?'
    }
    assert trace_data.route == 'knowledge_base'
    assert trace_data.search_required is True


async def test_non_factual_greeting_without_tool_call_stays_direct():
    chat_model = chat_model_with_handler(lambda request: _direct_response('Привет!'), streaming=False)
    node = create_analyze_intent_node(chat_model, vera_rag_kb, send_consultation_email, _CONTEXT_SETTINGS)

    trace_data = AgentRequestTraceData()
    token = set_request_trace(trace_data)
    try:
        result = await node(_state('привет'))
    finally:
        reset_request_trace(token)

    assert result == {}
    assert trace_data.route == 'direct'
