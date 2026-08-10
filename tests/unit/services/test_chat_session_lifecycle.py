from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from langgraph.checkpoint.redis.aio import AsyncRedisSaver

from app.db.models.chat_session import ChatSession
from app.exceptions.chat_session import (
    ChatSessionAccessDeniedError,
    ChatSessionResolutionConflictError,
)
from app.repositories.chat_session import ChatSessionRepository
from app.repositories.chat_turn import (
    ChatSessionTurnState,
    ChatTurnRepository,
)
from app.services.chat_session_access import (
    PREVIOUS_ANONYMOUS_HASH_METADATA_KEY,
    ensure_chat_session_access,
)
from app.services.chat_session_lifecycle import (
    ANONYMOUS_RECOVERY_OPERATION_ID_METADATA_KEY,
    SUCCESSOR_RECOVERY_DEADLINE_METADATA_KEY,
    SUCCESSOR_RECOVERY_PREDECESSOR_ID_METADATA_KEY,
    SUCCESSOR_RECOVERY_WINDOW_SECONDS,
    SUCCESSOR_SESSION_ID_METADATA_KEY,
    ChatSessionLifecycleService,
)


def _service(
    session_repository: AsyncMock,
    turn_repository: AsyncMock,
    checkpointer: AsyncMock,
) -> ChatSessionLifecycleService:
    return ChatSessionLifecycleService(
        chat_session_repository=session_repository,
        chat_turn_repository=turn_repository,
        checkpointer=checkpointer,
        session_ttl_seconds=86400,
    )


@pytest.mark.asyncio
async def test_resolve_session_creates_missing_requested_session():
    current = ChatSession(
        id=uuid4(),
        session_id='session-1',
        anonymous_token_hash='a' * 64,
        service_metadata={},
        last_activity_at=datetime.now(UTC),
    )
    session_repository = AsyncMock(spec=ChatSessionRepository)
    turn_repository = AsyncMock(spec=ChatTurnRepository)
    checkpointer = AsyncMock(spec=AsyncRedisSaver)
    session_repository.lock_or_create_for_lifecycle.return_value = (
        current,
        True,
    )

    result = await _service(
        session_repository,
        turn_repository,
        checkpointer,
    ).resolve_session(
        session_id='session-1',
        replacement_session_id='session-2',
        user_id=None,
        anonymous_token_hash='a' * 64,
        refreshed_anonymous_token_hash='b' * 64,
        replacement_anonymous_token_hash='c' * 64,
    )

    assert result.session_id == 'session-1'
    assert result.previous_session_id is None
    assert result.boundary == 'created'
    assert result.session_ttl_seconds == 86400
    session_repository.commit_lifecycle_changes.assert_awaited_once_with()
    turn_repository.get_session_turn_state.assert_not_awaited()


@pytest.mark.asyncio
async def test_resolve_session_rolls_anonymous_hash_after_current_proof():
    old_hash = 'a' * 64
    first_refreshed_hash = 'b' * 64
    different_refreshed_hash = 'c' * 64
    next_refreshed_hash = 'e' * 64
    current = ChatSession(
        id=uuid4(),
        session_id='session-1',
        anonymous_token_hash=old_hash,
        service_metadata={},
        last_activity_at=datetime.now(UTC),
    )
    session_repository = AsyncMock(spec=ChatSessionRepository)
    turn_repository = AsyncMock(spec=ChatTurnRepository)
    checkpointer = AsyncMock(spec=AsyncRedisSaver)
    session_repository.lock_or_create_for_lifecycle.return_value = (
        current,
        False,
    )
    turn_repository.get_session_turn_state.return_value = (
        ChatSessionTurnState(
            has_turns=False,
            has_live_processing_turn=False,
        )
    )
    service = _service(session_repository, turn_repository, checkpointer)

    first = await service.resolve_session(
        session_id='session-1',
        replacement_session_id='session-2',
        user_id='user-1',
        anonymous_token_hash=old_hash,
        refreshed_anonymous_token_hash=first_refreshed_hash,
        replacement_anonymous_token_hash='d' * 64,
    )
    with pytest.raises(ChatSessionAccessDeniedError):
        await service.resolve_session(
            session_id='session-1',
            replacement_session_id='session-3',
            user_id='user-1',
            anonymous_token_hash=old_hash,
            refreshed_anonymous_token_hash=first_refreshed_hash,
            replacement_anonymous_token_hash='d' * 64,
        )
    with pytest.raises(ChatSessionAccessDeniedError):
        await service.resolve_session(
            session_id='session-1',
            replacement_session_id='session-2',
            user_id='user-1',
            anonymous_token_hash=old_hash,
            refreshed_anonymous_token_hash=different_refreshed_hash,
            replacement_anonymous_token_hash='d' * 64,
        )
    retry = await service.resolve_session(
        session_id='session-1',
        replacement_session_id='session-2',
        user_id='user-1',
        anonymous_token_hash=old_hash,
        refreshed_anonymous_token_hash=first_refreshed_hash,
        replacement_anonymous_token_hash='d' * 64,
    )

    assert first.boundary == retry.boundary == 'retained'
    assert current.user_id is None
    assert current.anonymous_token_hash == first_refreshed_hash
    assert (
        current.service_metadata[PREVIOUS_ANONYMOUS_HASH_METADATA_KEY]
        == old_hash
    )
    assert (
        current.service_metadata[
            ANONYMOUS_RECOVERY_OPERATION_ID_METADATA_KEY
        ]
        == 'session-2'
    )
    ensure_chat_session_access(
        current,
        user_id=None,
        anonymous_token_hash=old_hash,
    )
    ensure_chat_session_access(
        current,
        user_id=None,
        anonymous_token_hash=first_refreshed_hash,
    )
    with pytest.raises(ChatSessionAccessDeniedError):
        ensure_chat_session_access(
            current,
            user_id=None,
            anonymous_token_hash=different_refreshed_hash,
        )

    next_resolution = await service.resolve_session(
        session_id='session-1',
        replacement_session_id='session-3',
        user_id='user-1',
        anonymous_token_hash=first_refreshed_hash,
        refreshed_anonymous_token_hash=next_refreshed_hash,
        replacement_anonymous_token_hash='d' * 64,
    )

    assert next_resolution.boundary == 'retained'
    assert current.anonymous_token_hash == next_refreshed_hash
    assert (
        current.service_metadata[PREVIOUS_ANONYMOUS_HASH_METADATA_KEY]
        == first_refreshed_hash
    )
    assert (
        current.service_metadata[
            ANONYMOUS_RECOVERY_OPERATION_ID_METADATA_KEY
        ]
        == 'session-3'
    )
    with pytest.raises(ChatSessionAccessDeniedError):
        ensure_chat_session_access(
            current,
            user_id=None,
            anonymous_token_hash=old_hash,
        )
    ensure_chat_session_access(
        current,
        user_id=None,
        anonymous_token_hash=first_refreshed_hash,
    )


@pytest.mark.asyncio
async def test_resolve_session_expires_without_checkpoint_or_live_lease():
    current = ChatSession(
        id=uuid4(),
        session_id='session-1',
        user_id='user-1',
        service_metadata={},
        last_activity_at=datetime.now(UTC),
    )
    successor = ChatSession(
        id=uuid4(),
        session_id='session-2',
        user_id='user-1',
        service_metadata={},
        last_activity_at=datetime.now(UTC),
    )
    session_repository = AsyncMock(spec=ChatSessionRepository)
    turn_repository = AsyncMock(spec=ChatTurnRepository)
    checkpointer = AsyncMock(spec=AsyncRedisSaver)
    session_repository.lock_or_create_for_lifecycle.side_effect = [
        (current, False),
        (successor, True),
    ]
    turn_repository.get_session_turn_state.return_value = (
        ChatSessionTurnState(
            has_turns=True,
            has_live_processing_turn=False,
        )
    )
    checkpointer.aget_tuple.return_value = None

    result = await _service(
        session_repository,
        turn_repository,
        checkpointer,
    ).resolve_session(
        session_id='session-1',
        replacement_session_id='session-2',
        user_id='user-1',
        anonymous_token_hash=None,
        refreshed_anonymous_token_hash=None,
        replacement_anonymous_token_hash=None,
    )

    assert result.session_id == 'session-2'
    assert result.previous_session_id == 'session-1'
    assert result.boundary == 'expired'
    assert current.closed_at is not None
    assert (
        current.service_metadata[SUCCESSOR_SESSION_ID_METADATA_KEY]
        == 'session-2'
    )
    checkpointer.aget_tuple.assert_awaited_once_with(
        {'configurable': {'thread_id': 'session-1'}}
    )


@pytest.mark.asyncio
async def test_resolve_session_retains_live_processing_before_first_checkpoint():
    current = ChatSession(
        id=uuid4(),
        session_id='session-processing',
        user_id='user-1',
        service_metadata={},
        last_activity_at=datetime.now(UTC),
    )
    session_repository = AsyncMock(spec=ChatSessionRepository)
    turn_repository = AsyncMock(spec=ChatTurnRepository)
    checkpointer = AsyncMock(spec=AsyncRedisSaver)
    session_repository.lock_or_create_for_lifecycle.return_value = (
        current,
        False,
    )
    turn_repository.get_session_turn_state.return_value = (
        ChatSessionTurnState(
            has_turns=True,
            has_live_processing_turn=True,
        )
    )
    checkpointer.aget_tuple.return_value = None

    result = await _service(
        session_repository,
        turn_repository,
        checkpointer,
    ).resolve_session(
        session_id='session-processing',
        replacement_session_id='session-replacement',
        user_id='user-1',
        anonymous_token_hash=None,
        refreshed_anonymous_token_hash=None,
        replacement_anonymous_token_hash=None,
    )

    assert result.boundary == 'retained'
    assert result.session_id == 'session-processing'
    assert current.closed_at is None
    checkpointer.aget_tuple.assert_awaited_once_with(
        {'configurable': {'thread_id': 'session-processing'}}
    )


@pytest.mark.asyncio
async def test_resolve_session_refreshes_live_redis_checkpoint_on_read():
    current = ChatSession(
        id=uuid4(),
        session_id='session-1',
        user_id='user-1',
        service_metadata={},
        last_activity_at=datetime.now(UTC),
    )
    session_repository = AsyncMock(spec=ChatSessionRepository)
    turn_repository = AsyncMock(spec=ChatTurnRepository)
    checkpointer = AsyncMock(spec=AsyncRedisSaver)
    session_repository.lock_or_create_for_lifecycle.return_value = (
        current,
        False,
    )
    turn_repository.get_session_turn_state.return_value = (
        ChatSessionTurnState(
            has_turns=True,
            has_live_processing_turn=False,
        )
    )
    checkpointer.aget_tuple.return_value = object()

    result = await _service(
        session_repository,
        turn_repository,
        checkpointer,
    ).resolve_session(
        session_id='session-1',
        replacement_session_id='session-2',
        user_id='user-1',
        anonymous_token_hash=None,
        refreshed_anonymous_token_hash=None,
        replacement_anonymous_token_hash=None,
    )

    assert result.boundary == 'retained'
    checkpointer.aget_tuple.assert_awaited_once_with(
        {'configurable': {'thread_id': 'session-1'}}
    )


@pytest.mark.asyncio
async def test_get_current_user_session_returns_db_stale_candidate_unchanged():
    current = ChatSession(
        id=uuid4(),
        session_id='session-1',
        user_id='user-1',
        service_metadata={},
        last_activity_at=datetime.now(UTC) - timedelta(days=2),
    )
    session_repository = AsyncMock(spec=ChatSessionRepository)
    turn_repository = AsyncMock(spec=ChatTurnRepository)
    checkpointer = AsyncMock(spec=AsyncRedisSaver)
    session_repository.get_current_by_user_id.return_value = current

    result = await _service(
        session_repository,
        turn_repository,
        checkpointer,
    ).get_current_user_session('user-1')

    assert result is current
    assert current.closed_at is None
    session_repository.commit_lifecycle_changes.assert_not_awaited()
    checkpointer.aget_tuple.assert_not_awaited()


@pytest.mark.asyncio
async def test_get_current_user_session_returns_missing_checkpoint_candidate():
    last_activity_at = datetime.now(UTC) - timedelta(minutes=1)
    current = ChatSession(
        id=uuid4(),
        session_id='session-1',
        user_id='user-1',
        service_metadata={},
        last_activity_at=last_activity_at,
    )
    session_repository = AsyncMock(spec=ChatSessionRepository)
    turn_repository = AsyncMock(spec=ChatTurnRepository)
    checkpointer = AsyncMock(spec=AsyncRedisSaver)
    session_repository.get_current_by_user_id.return_value = current
    turn_repository.get_session_turn_state.return_value = (
        ChatSessionTurnState(
            has_turns=True,
            has_live_processing_turn=False,
        )
    )
    checkpointer.aget_tuple.return_value = None

    result = await _service(
        session_repository,
        turn_repository,
        checkpointer,
    ).get_current_user_session('user-1')

    assert result is current
    assert current.closed_at is None
    assert current.last_activity_at == last_activity_at
    session_repository.commit_lifecycle_changes.assert_not_awaited()
    checkpointer.aget_tuple.assert_awaited_once_with(
        {'configurable': {'thread_id': 'session-1'}}
    )


@pytest.mark.asyncio
async def test_get_current_user_session_refreshes_active_candidate():
    last_activity_at = datetime.now(UTC) - timedelta(minutes=1)
    current = ChatSession(
        id=uuid4(),
        session_id='session-1',
        user_id='user-1',
        service_metadata={},
        last_activity_at=last_activity_at,
    )
    session_repository = AsyncMock(spec=ChatSessionRepository)
    turn_repository = AsyncMock(spec=ChatTurnRepository)
    checkpointer = AsyncMock(spec=AsyncRedisSaver)
    session_repository.get_current_by_user_id.return_value = current
    turn_repository.get_session_turn_state.return_value = (
        ChatSessionTurnState(
            has_turns=True,
            has_live_processing_turn=False,
        )
    )
    checkpointer.aget_tuple.return_value = object()

    result = await _service(
        session_repository,
        turn_repository,
        checkpointer,
    ).get_current_user_session('user-1')

    assert result is current
    assert current.last_activity_at > last_activity_at
    session_repository.commit_lifecycle_changes.assert_awaited_once_with()
    checkpointer.aget_tuple.assert_awaited_once_with(
        {'configurable': {'thread_id': 'session-1'}}
    )


@pytest.mark.asyncio
async def test_resolve_session_rejects_different_successor_on_retry():
    current = ChatSession(
        id=uuid4(),
        session_id='session-1',
        user_id='user-1',
        service_metadata={
            SUCCESSOR_SESSION_ID_METADATA_KEY: 'session-2',
        },
        last_activity_at=datetime.now(UTC) - timedelta(days=2),
        closed_at=datetime.now(UTC),
    )
    session_repository = AsyncMock(spec=ChatSessionRepository)
    turn_repository = AsyncMock(spec=ChatTurnRepository)
    checkpointer = AsyncMock(spec=AsyncRedisSaver)
    session_repository.lock_or_create_for_lifecycle.return_value = (
        current,
        False,
    )
    service = _service(session_repository, turn_repository, checkpointer)

    with pytest.raises(ChatSessionResolutionConflictError):
        await service.resolve_session(
            session_id='session-1',
            replacement_session_id='session-3',
            user_id='user-1',
            anonymous_token_hash=None,
            refreshed_anonymous_token_hash=None,
            replacement_anonymous_token_hash=None,
        )


@pytest.mark.asyncio
async def test_expired_retry_requires_exact_persisted_anonymous_hashes():
    old_hash = 'a' * 64
    first_replacement_hash = 'b' * 64
    retry_replacement_hash = 'c' * 64
    current = ChatSession(
        id=uuid4(),
        session_id='session-1',
        anonymous_token_hash=old_hash,
        service_metadata={},
        last_activity_at=datetime.now(UTC) - timedelta(days=2),
    )
    successor = ChatSession(
        id=uuid4(),
        session_id='session-2',
        anonymous_token_hash=first_replacement_hash,
        service_metadata={},
        last_activity_at=datetime.now(UTC),
    )
    session_repository = AsyncMock(spec=ChatSessionRepository)
    turn_repository = AsyncMock(spec=ChatTurnRepository)
    checkpointer = AsyncMock(spec=AsyncRedisSaver)
    session_repository.lock_or_create_for_lifecycle.side_effect = [
        (current, False),
        (successor, True),
        (current, False),
        (current, False),
        (current, False),
    ]
    session_repository.lock_by_session_id.return_value = successor
    service = _service(session_repository, turn_repository, checkpointer)

    first = await service.resolve_session(
        session_id='session-1',
        replacement_session_id='session-2',
        user_id=None,
        anonymous_token_hash=old_hash,
        refreshed_anonymous_token_hash='d' * 64,
        replacement_anonymous_token_hash=first_replacement_hash,
    )
    with pytest.raises(ChatSessionAccessDeniedError):
        await service.resolve_session(
            session_id='session-1',
            replacement_session_id='session-2',
            user_id=None,
            anonymous_token_hash=old_hash,
            refreshed_anonymous_token_hash='e' * 64,
            replacement_anonymous_token_hash=first_replacement_hash,
        )
    with pytest.raises(ChatSessionAccessDeniedError):
        await service.resolve_session(
            session_id='session-1',
            replacement_session_id='session-2',
            user_id=None,
            anonymous_token_hash=old_hash,
            refreshed_anonymous_token_hash='d' * 64,
            replacement_anonymous_token_hash=retry_replacement_hash,
        )
    retry = await service.resolve_session(
        session_id='session-1',
        replacement_session_id='session-2',
        user_id=None,
        anonymous_token_hash=old_hash,
        refreshed_anonymous_token_hash='d' * 64,
        replacement_anonymous_token_hash=first_replacement_hash,
    )

    assert first.boundary == retry.boundary == 'expired'
    assert first.session_id == retry.session_id == 'session-2'
    assert current.anonymous_token_hash == 'd' * 64
    assert (
        current.service_metadata[PREVIOUS_ANONYMOUS_HASH_METADATA_KEY]
        == old_hash
    )
    assert (
        current.service_metadata[
            ANONYMOUS_RECOVERY_OPERATION_ID_METADATA_KEY
        ]
        == 'session-2'
    )
    assert successor.anonymous_token_hash == first_replacement_hash
    assert (
        successor.service_metadata[
            SUCCESSOR_RECOVERY_PREDECESSOR_ID_METADATA_KEY
        ]
        == 'session-1'
    )
    assert (
        successor.service_metadata[
            SUCCESSOR_RECOVERY_DEADLINE_METADATA_KEY
        ]
        > datetime.now(UTC).timestamp()
    )
    ensure_chat_session_access(
        successor,
        user_id=None,
        anonymous_token_hash=first_replacement_hash,
    )
    with pytest.raises(ChatSessionAccessDeniedError):
        ensure_chat_session_access(
            successor,
            user_id=None,
            anonymous_token_hash=retry_replacement_hash,
        )


@pytest.mark.asyncio
async def test_expired_retry_rejects_foreign_auth_successor():
    current = ChatSession(
        id=uuid4(),
        session_id='session-1',
        anonymous_token_hash='a' * 64,
        service_metadata={
            SUCCESSOR_SESSION_ID_METADATA_KEY: 'session-2',
        },
        last_activity_at=datetime.now(UTC) - timedelta(days=2),
        closed_at=datetime.now(UTC),
    )
    successor = ChatSession(
        id=uuid4(),
        session_id='session-2',
        user_id='owner-2',
        service_metadata={},
        last_activity_at=datetime.now(UTC),
    )
    session_repository = AsyncMock(spec=ChatSessionRepository)
    turn_repository = AsyncMock(spec=ChatTurnRepository)
    checkpointer = AsyncMock(spec=AsyncRedisSaver)
    session_repository.lock_or_create_for_lifecycle.return_value = (
        current,
        False,
    )
    session_repository.lock_by_session_id.return_value = successor
    service = _service(session_repository, turn_repository, checkpointer)

    with pytest.raises(ChatSessionAccessDeniedError):
        await service.resolve_session(
            session_id='session-1',
            replacement_session_id='session-2',
            user_id='owner-1',
            anonymous_token_hash='a' * 64,
            refreshed_anonymous_token_hash='b' * 64,
            replacement_anonymous_token_hash='c' * 64,
        )

    assert successor.user_id == 'owner-2'


@pytest.mark.asyncio
async def test_normal_successor_resolve_closes_predecessor_recovery():
    predecessor = ChatSession(
        id=uuid4(),
        session_id='session-1',
        anonymous_token_hash='a' * 64,
        service_metadata={
            SUCCESSOR_SESSION_ID_METADATA_KEY: 'session-2',
        },
        last_activity_at=datetime.now(UTC) - timedelta(days=2),
        closed_at=datetime.now(UTC),
    )
    successor = ChatSession(
        id=uuid4(),
        session_id='session-2',
        anonymous_token_hash='b' * 64,
        service_metadata={
            SUCCESSOR_RECOVERY_PREDECESSOR_ID_METADATA_KEY: 'session-1',
            SUCCESSOR_RECOVERY_DEADLINE_METADATA_KEY: (
                datetime.now(UTC).timestamp()
                + SUCCESSOR_RECOVERY_WINDOW_SECONDS
            ),
        },
        last_activity_at=datetime.now(UTC),
    )
    session_repository = AsyncMock(spec=ChatSessionRepository)
    turn_repository = AsyncMock(spec=ChatTurnRepository)
    checkpointer = AsyncMock(spec=AsyncRedisSaver)
    session_repository.lock_or_create_for_lifecycle.side_effect = [
        (successor, False),
        (predecessor, False),
    ]
    session_repository.lock_by_session_id.return_value = successor
    turn_repository.get_session_turn_state.return_value = (
        ChatSessionTurnState(
            has_turns=False,
            has_live_processing_turn=False,
        )
    )
    service = _service(session_repository, turn_repository, checkpointer)

    normal = await service.resolve_session(
        session_id='session-2',
        replacement_session_id='session-3',
        user_id=None,
        anonymous_token_hash='b' * 64,
        refreshed_anonymous_token_hash='c' * 64,
        replacement_anonymous_token_hash='d' * 64,
    )
    with pytest.raises(ChatSessionAccessDeniedError):
        await service.resolve_session(
            session_id='session-1',
            replacement_session_id='session-2',
            user_id=None,
            anonymous_token_hash='a' * 64,
            refreshed_anonymous_token_hash='e' * 64,
            replacement_anonymous_token_hash='f' * 64,
        )

    assert normal.boundary == 'retained'
    assert (
        SUCCESSOR_RECOVERY_PREDECESSOR_ID_METADATA_KEY
        not in successor.service_metadata
    )
    assert (
        SUCCESSOR_RECOVERY_DEADLINE_METADATA_KEY
        not in successor.service_metadata
    )


@pytest.mark.asyncio
async def test_expired_successor_recovery_rejects_expired_deadline():
    predecessor = ChatSession(
        id=uuid4(),
        session_id='session-1',
        anonymous_token_hash='a' * 64,
        service_metadata={
            SUCCESSOR_SESSION_ID_METADATA_KEY: 'session-2',
        },
        last_activity_at=datetime.now(UTC) - timedelta(days=2),
        closed_at=datetime.now(UTC),
    )
    successor = ChatSession(
        id=uuid4(),
        session_id='session-2',
        anonymous_token_hash='b' * 64,
        service_metadata={
            SUCCESSOR_RECOVERY_PREDECESSOR_ID_METADATA_KEY: 'session-1',
            SUCCESSOR_RECOVERY_DEADLINE_METADATA_KEY: (
                datetime.now(UTC).timestamp() - 1
            ),
        },
        last_activity_at=datetime.now(UTC),
    )
    session_repository = AsyncMock(spec=ChatSessionRepository)
    turn_repository = AsyncMock(spec=ChatTurnRepository)
    checkpointer = AsyncMock(spec=AsyncRedisSaver)
    session_repository.lock_or_create_for_lifecycle.return_value = (
        predecessor,
        False,
    )
    session_repository.lock_by_session_id.return_value = successor
    service = _service(session_repository, turn_repository, checkpointer)

    with pytest.raises(ChatSessionAccessDeniedError):
        await service.resolve_session(
            session_id='session-1',
            replacement_session_id='session-2',
            user_id=None,
            anonymous_token_hash='a' * 64,
            refreshed_anonymous_token_hash='c' * 64,
            replacement_anonymous_token_hash='d' * 64,
        )

    assert successor.anonymous_token_hash == 'b' * 64
