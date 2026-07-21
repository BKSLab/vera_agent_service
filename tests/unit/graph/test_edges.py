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


def test_routes_to_generate_direct_when_no_tool_calls_were_added():
    """analyze_intent не изменяет messages, если тул не нужен — последним
    остаётся исходный HumanMessage, не AIMessage."""
    state = _state([HumanMessage(content='привет')])
    assert route_after_analyze_intent(state) == 'generate_direct'


def test_routes_to_generate_direct_when_ai_message_has_empty_tool_calls():
    ai_message = AIMessage(content='Привет! Чем помочь?', tool_calls=[])
    state = _state([HumanMessage(content='привет'), ai_message])
    assert route_after_analyze_intent(state) == 'generate_direct'
