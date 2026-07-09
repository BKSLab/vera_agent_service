"""Интеграционные тесты Redis checkpointer'а (Этап 5) на реальном Redis
Stack из `docker-compose.yml` (сервис `redis` — `redis/redis-stack-server`,
не ванильный `redis`, см. docstring `app/checkpoint/redis_saver.py`).
"""

import asyncio
import uuid

import pytest
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph

from app.checkpoint.redis_saver import get_redis_checkpointer
from app.core.settings import RedisSettings
from app.graph.state import AgentState

pytestmark = pytest.mark.integration


def _echo_node(state: AgentState) -> dict:
    return {}


def _build_echo_graph():
    builder = StateGraph(AgentState)
    builder.add_node('echo', _echo_node)
    builder.set_entry_point('echo')
    builder.set_finish_point('echo')
    return builder


def _initial_state(text: str) -> dict:
    return {
        'session_id': 'redis-test',
        'user_id': None,
        'messages': [HumanMessage(content=text)],
        'retrieved_chunks': [],
        'tool_calls': [],
        'search_unavailable': False,
    }


async def test_conversation_history_persists_via_real_redis():
    """`thread_id` — заново сгенерированный UUID на каждый запуск теста:
    Redis реальный и персистентный (не поднимается/уничтожается на каждый
    тест, в отличие от мок MCP-сервера), фиксированная строка накапливала
    бы сообщения от предыдущих прогонов теста и ломала бы проверку —
    найдено эмпирически при повторных запусках во время отладки этого
    файла."""
    settings = RedisSettings(redis_host='localhost', redis_port=6379, redis_session_ttl_seconds=86400)
    async with get_redis_checkpointer(settings) as checkpointer:
        graph = _build_echo_graph().compile(checkpointer=checkpointer)
        config = {'configurable': {'thread_id': str(uuid.uuid4())}}

        await graph.ainvoke(_initial_state('Первый вопрос'), config=config)
        result = await graph.ainvoke(_initial_state('Второй вопрос'), config=config)

        contents = [message.content for message in result['messages']]
        assert contents == ['Первый вопрос', 'Второй вопрос']


async def test_sessions_are_isolated_via_real_redis():
    settings = RedisSettings(redis_host='localhost', redis_port=6379, redis_session_ttl_seconds=86400)
    async with get_redis_checkpointer(settings) as checkpointer:
        graph = _build_echo_graph().compile(checkpointer=checkpointer)
        thread_id_a, thread_id_b = str(uuid.uuid4()), str(uuid.uuid4())

        await graph.ainvoke(
            _initial_state('Вопрос в сессии A'), config={'configurable': {'thread_id': thread_id_a}}
        )
        result_b = await graph.ainvoke(
            _initial_state('Вопрос в сессии B'), config={'configurable': {'thread_id': thread_id_b}}
        )

        contents = [message.content for message in result_b['messages']]
        assert contents == ['Вопрос в сессии B']


async def test_session_expires_after_ttl_of_inactivity():
    """TTL = 2 секунды (0.033 мин — минимальная осмысленная гранулярность,
    библиотека переводит в целые секунды). После TTL без обращений сессия
    стартует заново, без истории."""
    settings = RedisSettings(redis_host='localhost', redis_port=6379, redis_session_ttl_seconds=2)
    async with get_redis_checkpointer(settings) as checkpointer:
        graph = _build_echo_graph().compile(checkpointer=checkpointer)
        config = {'configurable': {'thread_id': str(uuid.uuid4())}}

        await graph.ainvoke(_initial_state('Вопрос до истечения TTL'), config=config)
        await asyncio.sleep(3.5)
        result = await graph.ainvoke(_initial_state('Вопрос после истечения TTL'), config=config)

        contents = [message.content for message in result['messages']]
        assert contents == ['Вопрос после истечения TTL']
