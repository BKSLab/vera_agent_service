"""Аренда реплики на реальном PostgreSQL.

Проверяется то, ради чего аренда и вводилась: после падения процесса запись
остаётся `processing`, и без различения живой и просроченной аренды повторная
доставка того же `request_id` отбрасывалась бы как дубликат вместе с потерей
запроса (VERA-014).
"""

from datetime import UTC, datetime, timedelta

import pytest

from app.db.models.chat_session import ChatSession
from app.db.models.chat_turn import (
    STATUS_DELIVERY_UNCONFIRMED,
    STATUS_PROCESSING,
    ChatTurn,
)
from app.repositories.chat_session import ChatSessionRepository
from app.repositories.chat_turn import ChatTurnRepository

pytestmark = pytest.mark.integration


async def _create_turn(db_session, *, lease_until, request_id='request-1') -> ChatTurn:
    session_repository = ChatSessionRepository(db_session)
    turn_repository = ChatTurnRepository(db_session)
    chat_session = await session_repository.save(
        ChatSession(session_id='session-1', anonymous_token_hash='a' * 64)
    )
    return await turn_repository.save(
        ChatTurn(
            request_id=request_id,
            chat_session_id=chat_session.id,
            sequence_number=1,
            question='Вопрос',
            status=STATUS_PROCESSING,
            lease_until=lease_until,
            attempt_count=1,
            worker_id='worker-a',
        )
    )


async def test_expired_lease_is_reclaimed_by_another_worker(db_session):
    await _create_turn(db_session, lease_until=datetime.now(UTC) - timedelta(minutes=5))
    turn_repository = ChatTurnRepository(db_session)

    reclaimed = await turn_repository.claim_expired_lease(
        request_id='request-1',
        worker_id='worker-b',
        lease_seconds=900.0,
    )

    assert reclaimed is True
    refreshed = await turn_repository.get_by_request_id('request-1')
    assert refreshed.worker_id == 'worker-b'
    assert refreshed.attempt_count == 2
    assert refreshed.lease_until > datetime.now(UTC)


async def test_live_lease_is_not_reclaimed(db_session):
    await _create_turn(db_session, lease_until=datetime.now(UTC) + timedelta(minutes=5))
    turn_repository = ChatTurnRepository(db_session)

    reclaimed = await turn_repository.claim_expired_lease(
        request_id='request-1',
        worker_id='worker-b',
        lease_seconds=900.0,
    )

    assert reclaimed is False
    refreshed = await turn_repository.get_by_request_id('request-1')
    assert refreshed.worker_id == 'worker-a'
    assert refreshed.attempt_count == 1


async def test_only_one_worker_wins_the_same_expired_lease(db_session):
    """Проверка и захват аренды выполняются одним UPDATE, а не двумя шагами."""
    await _create_turn(db_session, lease_until=datetime.now(UTC) - timedelta(minutes=5))
    turn_repository = ChatTurnRepository(db_session)

    first = await turn_repository.claim_expired_lease(
        request_id='request-1', worker_id='worker-b', lease_seconds=900.0
    )
    second = await turn_repository.claim_expired_lease(
        request_id='request-1', worker_id='worker-c', lease_seconds=900.0
    )

    assert first is True
    assert second is False


async def test_stale_turn_is_closed_with_unconfirmed_outcome(db_session):
    """Брошенная реплика закрывается, а не висит `processing` вечно."""
    await _create_turn(db_session, lease_until=datetime.now(UTC) - timedelta(hours=2))
    turn_repository = ChatTurnRepository(db_session)

    closed = await turn_repository.reconcile_stale(
        stale_before=datetime.now(UTC) - timedelta(minutes=30),
        detail='Обработка запроса была прервана.',
    )

    assert closed == 1
    refreshed = await turn_repository.get_by_request_id('request-1')
    assert refreshed.status == STATUS_DELIVERY_UNCONFIRMED
    assert refreshed.terminal_detail == 'Обработка запроса была прервана.'
    assert refreshed.lease_until is None


async def test_recent_lease_is_not_closed_by_reconciliation(db_session):
    """Очистка не должна обгонять штатную повторную доставку после рестарта."""
    await _create_turn(db_session, lease_until=datetime.now(UTC) - timedelta(minutes=1))
    turn_repository = ChatTurnRepository(db_session)

    closed = await turn_repository.reconcile_stale(
        stale_before=datetime.now(UTC) - timedelta(minutes=30),
        detail='Обработка запроса была прервана.',
    )

    assert closed == 0
    refreshed = await turn_repository.get_by_request_id('request-1')
    assert refreshed.status == STATUS_PROCESSING
