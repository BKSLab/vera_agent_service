import json

import httpx
import pytest
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import tool

from app.core.settings import GraphContextSettings
from app.graph.nodes.analyze_intent import create_analyze_intent_node
from app.graph.policy import (
    SIMPLIFY_ANSWER_REQUEST,
    contains_legal_reference,
    is_probably_factual_or_legal_question,
    is_reference_only,
)
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
    assert trace_data.route_reason == 'model_tool_call'
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
    assert trace_data.route_reason == 'model_direct'
    assert trace_data.search_required is False


async def test_consultation_email_tool_call_is_allowed_for_explicit_request_with_email():
    """Решение модели вызвать email-тул передаётся в граф без второго guard."""
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
    assert trace_data.route_reason == 'model_tool_call'


async def test_reference_inside_email_command_does_not_bypass_llm_tool_choice():
    query = 'Отправь консультацию по статье 81 ТК РФ на user@example.com'
    arguments = {
        'consultation_text': 'Консультация по статье 81 ТК РФ.',
        'email': 'user@example.com',
    }
    calls = {'count': 0}

    def handler(request):
        calls['count'] += 1
        return _tool_call_response('send_consultation_email', arguments)

    chat_model = chat_model_with_handler(handler, streaming=False)
    node = create_analyze_intent_node(
        chat_model,
        vera_rag_kb,
        send_consultation_email,
        _CONTEXT_SETTINGS,
    )

    result = await node(_state(query))

    tool_call = result['messages'][0].tool_calls[0]
    assert calls['count'] == 1
    assert tool_call['name'] == 'send_consultation_email'
    assert tool_call['args'] == arguments


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
    assert trace_data.route_reason == 'factual_or_legal_keyword'
    assert trace_data.search_required is True


def test_contains_legal_reference_accepts_only_strong_legal_requisites():
    references = (
        'п. 2 ч. 1 ст. 81 ТК РФ',
        'статья 81 ТК РФ',
        'статьи 81 ТК РФ',
        'статье 81 ТК РФ',
        'статью 128 ТК РФ',
        'ст. 128 ТК РФ',
        'ст 128 ТК РФ',
        'статьёй 128 ТК РФ',
        'ТК РФ, статья 21',
        'статья 21 № 181-ФЗ',
        'ФЗ-181',
        'Можно ли уволить по п. 2 ч. 1 ст. 81 ТК РФ?',
    )

    assert all(contains_legal_reference(text) for text in references)


def test_contains_legal_reference_rejects_ambiguous_everyday_phrases():
    phrases = (
        'Статья 21',
        'Статья 21 в журнале посвящена найму персонала',
        'Пункт 2 плана и часть 1 отчёта',
        'Мне 81 год',
        'Объясни предыдущий ответ проще',
    )

    assert not any(contains_legal_reference(text) for text in phrases)


@pytest.mark.parametrize(
    'query',
    (
        'п. 2 ч. 1 ст. 81 ТК РФ',
        'п. 1 ч. 1 ст. 999 ТК РФ',
        'статья 81 ТК РФ',
        'ТК РФ, статья 21',
        'ФЗ-181',
        'Статья 21 Федерального закона от 24.11.1995 № 181-ФЗ',
    ),
)
def test_reference_only_accepts_replica_made_entirely_of_legal_requisites(query):
    assert is_reference_only(query) is True


@pytest.mark.parametrize(
    'query',
    (
        'Можно ли уволить по п. 2 ч. 1 ст. 81 ТК РФ?',
        'Правда ли, что положено 90 дней отпуска по статье 128 ТК РФ?',
        'Отправь консультацию по статье 81 ТК РФ на user@example.com',
        'Статья 21',
        '',
    ),
)
def test_reference_only_rejects_questions_commands_and_incomplete_requisites(query):
    assert is_reference_only(query) is False


@pytest.mark.parametrize(
    'query',
    (
        'п. 2 ч. 1 ст. 81 ТК РФ',
        'п. 1 ч. 1 ст. 999 ТК РФ',
        'Статья 21 Федерального закона от 24.11.1995 № 181-ФЗ',
    ),
)
async def test_reference_only_bypasses_llm_and_goes_verbatim_into_kb_search(query):
    def fail_if_called(request):
        pytest.fail('reference_only не должен вызывать intent-LLM')

    chat_model = chat_model_with_handler(fail_if_called, streaming=False)
    node = create_analyze_intent_node(
        chat_model,
        vera_rag_kb,
        send_consultation_email,
        _CONTEXT_SETTINGS,
    )

    trace_data = AgentRequestTraceData()
    token = set_request_trace(trace_data)
    try:
        result = await node(_state(query))
    finally:
        reset_request_trace(token)

    tool_call = result['messages'][0].tool_calls[0]
    assert tool_call['name'] == 'vera_rag_kb'
    assert tool_call['args'] == {'query': query}
    assert trace_data.route == 'knowledge_base'
    assert trace_data.route_reason == 'reference_only'
    assert trace_data.search_required is True


@pytest.mark.parametrize(
    'query',
    (
        'Можно ли уволить по п. 2 ч. 1 ст. 81 ТК РФ?',
        'Правда ли, что работающему инвалиду положено 90 дней отпуска по статье 128 ТК РФ?',
    ),
)
async def test_question_with_legal_reference_still_calls_llm_before_guard(query):
    calls = {'count': 0}

    def handler(request):
        calls['count'] += 1
        return _direct_response('Отвечу без поиска.')

    chat_model = chat_model_with_handler(handler, streaming=False)
    node = create_analyze_intent_node(
        chat_model,
        vera_rag_kb,
        send_consultation_email,
        _CONTEXT_SETTINGS,
    )

    trace_data = AgentRequestTraceData()
    token = set_request_trace(trace_data)
    try:
        result = await node(_state(query))
    finally:
        reset_request_trace(token)

    tool_call = result['messages'][0].tool_calls[0]
    assert calls['count'] == 1
    assert tool_call['name'] == 'vera_rag_kb'
    assert tool_call['args'] == {'query': query}
    assert trace_data.route == 'knowledge_base'
    assert trace_data.route_reason == 'legal_reference'


async def test_article_term_without_strong_reference_uses_bounded_keyword_guard():
    query = 'Какая статья защищает беременных?'
    chat_model = chat_model_with_handler(
        lambda request: _direct_response('Отвечу без поиска.'),
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
        result = await node(_state(query))
    finally:
        reset_request_trace(token)

    tool_call = result['messages'][0].tool_calls[0]
    assert tool_call['name'] == 'vera_rag_kb'
    assert tool_call['args'] == {'query': query}
    assert trace_data.route == 'knowledge_base'
    assert trace_data.route_reason == 'factual_or_legal_keyword'


def test_article_word_guard_does_not_match_verb_stat():
    assert is_probably_factual_or_legal_question('Как стать увереннее?') is False


def test_keyword_guard_matches_stems_only_at_word_start():
    assert is_probably_factual_or_legal_question('Каким способом поменять пароль?') is False
    assert is_probably_factual_or_legal_question(
        'Каким способом работодатель должен выполнить квоту?'
    ) is True
    assert is_probably_factual_or_legal_question('Как оформить пособие?') is True


async def test_password_question_without_tool_call_is_not_forced_into_kb_search():
    chat_model = chat_model_with_handler(
        lambda request: _direct_response('Уточните, о каком сервисе идёт речь.'),
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
        result = await node(_state('Каким способом поменять пароль?'))
    finally:
        reset_request_trace(token)

    assert result == {}
    assert trace_data.route == 'direct'
    assert trace_data.route_reason == 'model_direct'


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
    assert trace_data.route_reason == 'model_direct'


def test_simplify_answer_request_is_not_treated_as_legal_question():
    """Текст кнопки «Объяснить проще» обязан оставаться чистым по ключевым
    словам VERA-021.

    Иначе нажатие кнопки под RAG-ответом код примет за новый правовой вопрос,
    принудительно уведёт в `vera_rag_kb` и вернёт второй ответ по базе знаний
    вместо переформулировки предыдущего.
    """
    assert is_probably_factual_or_legal_question(SIMPLIFY_ANSWER_REQUEST) is False


async def test_simplify_answer_request_stays_direct():
    """Просьба упростить ответ идёт в `generate_direct` и переформулирует
    предыдущий ответ по истории, а не переспрашивает базу знаний."""
    chat_model = chat_model_with_handler(
        lambda request: _direct_response('Если коротко: работодатель обязан.'),
        streaming=False,
    )
    node = create_analyze_intent_node(chat_model, vera_rag_kb, send_consultation_email, _CONTEXT_SETTINGS)

    trace_data = AgentRequestTraceData()
    token = set_request_trace(trace_data)
    try:
        result = await node(_state(SIMPLIFY_ANSWER_REQUEST))
    finally:
        reset_request_trace(token)

    assert result == {}
    assert trace_data.route == 'direct'
    assert trace_data.route_reason == 'model_direct'
