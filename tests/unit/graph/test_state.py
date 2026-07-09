from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph

from app.graph.state import AgentState


def _echo_node(state: AgentState) -> dict:
    return {}


def test_agent_state_has_expected_keys():
    assert set(AgentState.__annotations__) == {
        'session_id',
        'user_id',
        'messages',
        'retrieved_chunks',
        'tool_calls',
    }


async def test_messages_accumulate_via_add_messages_reducer():
    """Новое сообщение ДОПОЛНЯЕТ историю восстановленную checkpointer'ом,
    а не затирает её — ключевое условие решения "убрать history из payload
    RabbitMQ" (AGENT_SERVICE_PLAN.md, раздел 0.1). Без reducer'а
    `add_messages` в `AgentState.messages` этот тест падает: второй
    `ainvoke` вернул бы только последнее сообщение.
    """
    builder = StateGraph(AgentState)
    builder.add_node('echo', _echo_node)
    builder.set_entry_point('echo')
    builder.set_finish_point('echo')
    graph = builder.compile(checkpointer=InMemorySaver())

    config = {'configurable': {'thread_id': 'test-session'}}

    await graph.ainvoke(
        {
            'session_id': 'test-session',
            'user_id': None,
            'messages': [HumanMessage(content='Привет')],
            'retrieved_chunks': [],
            'tool_calls': [],
        },
        config=config,
    )
    result = await graph.ainvoke(
        {
            'session_id': 'test-session',
            'user_id': None,
            'messages': [HumanMessage(content='Второй вопрос')],
            'retrieved_chunks': [],
            'tool_calls': [],
        },
        config=config,
    )

    contents = [message.content for message in result['messages']]
    assert contents == ['Привет', 'Второй вопрос']


async def test_messages_do_not_leak_between_sessions():
    """Разные `thread_id` (session_id) — независимые истории."""
    builder = StateGraph(AgentState)
    builder.add_node('echo', _echo_node)
    builder.set_entry_point('echo')
    builder.set_finish_point('echo')
    graph = builder.compile(checkpointer=InMemorySaver())

    await graph.ainvoke(
        {
            'session_id': 'session-a',
            'user_id': None,
            'messages': [HumanMessage(content='Вопрос в сессии A')],
            'retrieved_chunks': [],
            'tool_calls': [],
        },
        config={'configurable': {'thread_id': 'session-a'}},
    )
    result_b = await graph.ainvoke(
        {
            'session_id': 'session-b',
            'user_id': None,
            'messages': [HumanMessage(content='Вопрос в сессии B')],
            'retrieved_chunks': [],
            'tool_calls': [],
        },
        config={'configurable': {'thread_id': 'session-b'}},
    )

    contents = [message.content for message in result_b['messages']]
    assert contents == ['Вопрос в сессии B']
