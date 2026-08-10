import pytest
from langchain_core.messages import AIMessage, HumanMessage

from app.graph.edges import route_after_analyze_intent


def _state(messages):
    return {
        'session_id': 's',
        'user_id': None,
        'messages': messages,
        'retrieved_chunks': [],
        'tool_calls': [],
        'search_unavailable': False,
    }


def test_routes_to_call_kb_search_when_tool_calls_present():
    ai_message = AIMessage(
        content='',
        tool_calls=[{'id': 'call_1', 'name': 'vera_rag_kb', 'args': {'query': 'квота'}}],
    )
    state = _state([HumanMessage(content='квота?'), ai_message])
    assert route_after_analyze_intent(state) == 'call_kb_search'


def test_routes_to_consultation_email_node():
    ai_message = AIMessage(
        content='',
        tool_calls=[
            {
                'id': 'call_2',
                'name': 'send_consultation_email',
                'args': {
                    'consultation_text': 'Текст консультации',
                    'email': 'user@example.com',
                },
            }
        ],
    )
    state = _state([HumanMessage(content='отправь'), ai_message])
    assert route_after_analyze_intent(state) == 'call_consultation_email'


def test_parallel_tool_calls_are_rejected():
    """Больше одного tool_call от модели — ошибка маршрутизации, а не
    повод молча выбрать первый (VERA-021)."""
    ai_message = AIMessage(
        content='',
        tool_calls=[
            {'id': 'call_1', 'name': 'vera_rag_kb', 'args': {'query': 'квота'}},
            {
                'id': 'call_2',
                'name': 'send_consultation_email',
                'args': {'consultation_text': 'текст', 'email': 'user@example.com'},
            },
        ],
    )
    state = _state([HumanMessage(content='вопрос'), ai_message])

    with pytest.raises(ValueError, match='параллельные вызовы'):
        route_after_analyze_intent(state)


def test_unknown_tool_call_is_rejected():
    ai_message = AIMessage(
        content='',
        tool_calls=[{'id': 'call_3', 'name': 'unknown_tool', 'args': {}}],
    )
    state = _state([HumanMessage(content='вызови'), ai_message])

    with pytest.raises(ValueError, match='unknown_tool'):
        route_after_analyze_intent(state)


def test_routes_to_generate_direct_when_no_tool_calls_were_added():
    """analyze_intent не изменяет messages, если тул не нужен — последним
    остаётся исходный HumanMessage, не AIMessage."""
    state = _state([HumanMessage(content='привет')])
    assert route_after_analyze_intent(state) == 'generate_direct'


def test_routes_to_generate_direct_when_ai_message_has_empty_tool_calls():
    ai_message = AIMessage(content='Привет! Чем помочь?', tool_calls=[])
    state = _state([HumanMessage(content='привет'), ai_message])
    assert route_after_analyze_intent(state) == 'generate_direct'
