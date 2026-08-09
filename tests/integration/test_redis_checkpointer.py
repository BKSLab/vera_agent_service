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
from app.messaging.consumer import _initial_state as _consumer_initial_state
from app.messaging.schemas import AgentRequestMessage
from app.repositories.chat_session import ChatSessionRepository
from app.repositories.chat_turn import ChatTurnRepository
from app.services.chat_persistence import START_CLAIMED, ChatPersistenceService

pytestmark = pytest.mark.integration


def _echo_node(state: AgentState) -> dict:
    return {}


def _build_echo_graph():
    builder = StateGraph(AgentState)
    builder.add_node('echo', _echo_node)
    builder.set_entry_point('echo')
    builder.set_finish_point('echo')
    return builder


def _build_flaky_graph():
    attempts = 0

    def flaky_node(state: AgentState) -> dict:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError('Сбой после сохранения initial checkpoint')
        return {}

    builder = StateGraph(AgentState)
    builder.add_node('flaky', flaky_node)
    builder.set_entry_point('flaky')
    builder.set_finish_point('flaky')
    return builder


async def _assert_failed_checkpoint_contains_original_message(
    checkpointer,
    config: dict,
    payload: AgentRequestMessage,
) -> None:
    checkpoint_tuple = await checkpointer.aget_tuple(config)
    assert checkpoint_tuple is not None
    messages = checkpoint_tuple.checkpoint['channel_values']['messages']
    human_messages = [
        message for message in messages if isinstance(message, HumanMessage)
    ]
    assert [(message.id, message.content) for message in human_messages] == [
        (payload.request_id, payload.message)
    ]


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


async def test_checkpointer_setup_fails_fast_when_redis_unreachable():
    """Тест устойчивости (Этап 10.6): Redis недоступен в момент установки
    checkpointer'а — падает с понятной ошибкой быстро, не зависает."""
    settings = RedisSettings(redis_host='localhost', redis_port=1, redis_session_ttl_seconds=86400)

    with pytest.raises(Exception):  # noqa: B017 - конкретный тип зависит от redis-py, важен сам факт быстрого отказа
        async with asyncio.timeout(10.0):
            async with get_redis_checkpointer(settings):
                pass


async def test_retry_same_request_does_not_duplicate_human_message_in_real_redis():
    settings = RedisSettings(
        redis_host='localhost',
        redis_port=6379,
        redis_session_ttl_seconds=86400,
    )
    payload = AgentRequestMessage(
        session_id=str(uuid.uuid4()),
        request_id=str(uuid.uuid4()),
        user_id=None,
        anonymous_token_hash='a' * 64,
        message='Вопрос с повторной попыткой',
    )
    config = {'configurable': {'thread_id': payload.session_id}}

    async with get_redis_checkpointer(settings) as checkpointer:
        graph = _build_flaky_graph().compile(checkpointer=checkpointer)
        with pytest.raises(RuntimeError, match='initial checkpoint'):
            await graph.ainvoke(_consumer_initial_state(payload), config=config)
        await _assert_failed_checkpoint_contains_original_message(
            checkpointer,
            config,
            payload,
        )

        result = await graph.ainvoke(_consumer_initial_state(payload), config=config)

    human_messages = [
        message for message in result['messages'] if isinstance(message, HumanMessage)
    ]
    assert [(message.id, message.content) for message in human_messages] == [
        (payload.request_id, payload.message)
    ]


async def test_stale_lease_retry_does_not_duplicate_human_message_in_real_redis(
    db_session,
):
    settings = RedisSettings(
        redis_host='localhost',
        redis_port=6379,
        redis_session_ttl_seconds=86400,
    )
    payload = AgentRequestMessage(
        session_id=str(uuid.uuid4()),
        request_id=str(uuid.uuid4()),
        user_id=None,
        anonymous_token_hash='a' * 64,
        message='Вопрос после перезахвата аренды',
    )
    persistence_service = ChatPersistenceService(
        ChatSessionRepository(db_session),
        ChatTurnRepository(db_session),
    )
    first_start = await persistence_service.start_turn(
        session_id=payload.session_id,
        request_id=payload.request_id,
        user_id=payload.user_id,
        anonymous_token_hash=payload.anonymous_token_hash,
        question=payload.message,
        worker_id='worker-before-crash',
        lease_seconds=0.05,
    )
    config = {'configurable': {'thread_id': payload.session_id}}

    async with get_redis_checkpointer(settings) as checkpointer:
        graph = _build_flaky_graph().compile(checkpointer=checkpointer)
        with pytest.raises(RuntimeError, match='initial checkpoint'):
            await graph.ainvoke(_consumer_initial_state(payload), config=config)
        await _assert_failed_checkpoint_contains_original_message(
            checkpointer,
            config,
            payload,
        )

        await asyncio.sleep(0.1)
        reclaimed_start = await persistence_service.start_turn(
            session_id=payload.session_id,
            request_id=payload.request_id,
            user_id=payload.user_id,
            anonymous_token_hash=payload.anonymous_token_hash,
            question=payload.message,
            worker_id='worker-after-crash',
            lease_seconds=900.0,
        )
        result = await graph.ainvoke(_consumer_initial_state(payload), config=config)

    human_messages = [
        message for message in result['messages'] if isinstance(message, HumanMessage)
    ]
    assert first_start.outcome == START_CLAIMED
    assert reclaimed_start.outcome == START_CLAIMED
    assert [(message.id, message.content) for message in human_messages] == [
        (payload.request_id, payload.message)
    ]
