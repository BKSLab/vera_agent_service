from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from langgraph.checkpoint.redis.aio import AsyncRedisSaver

from app.db.models.chat_session import ChatSession
from app.exceptions.chat_session import (
    ChatSessionAccessDeniedError,
    ChatSessionAlreadyClosedError,
    ChatSessionNotFoundError,
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
async def test_create_session_creates_missing_anonymous_session():
    chat_session = ChatSession(
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
        chat_session,
        True,
    )

    result = await _service(
        session_repository,
        turn_repository,
        checkpointer,
    ).create_session(
        session_id='session-1',
        user_id=None,
        anonymous_token_hash='a' * 64,
    )

    assert result.session_id == 'session-1'
    assert result.session_ttl_seconds == 86400
    session_repository.commit_lifecycle_changes.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_create_session_returns_owned_open_session_on_retry():
    chat_session = ChatSession(
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
        chat_session,
        False,
    )

    result = await _service(
        session_repository,
        turn_repository,
        checkpointer,
    ).create_session(
        session_id='session-1',
        user_id='user-1',
        anonymous_token_hash=None,
    )

    assert result.session_id == 'session-1'
    session_repository.commit_lifecycle_changes.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_create_session_rejects_owned_closed_session():
    chat_session = ChatSession(
        id=uuid4(),
        session_id='session-1',
        user_id='user-1',
        service_metadata={},
        last_activity_at=datetime.now(UTC),
        closed_at=datetime.now(UTC),
    )
    session_repository = AsyncMock(spec=ChatSessionRepository)
    turn_repository = AsyncMock(spec=ChatTurnRepository)
    checkpointer = AsyncMock(spec=AsyncRedisSaver)
    session_repository.lock_or_create_for_lifecycle.return_value = (
        chat_session,
        False,
    )

    with pytest.raises(ChatSessionAlreadyClosedError) as exc_info:
        await _service(
            session_repository,
            turn_repository,
            checkpointer,
        ).create_session(
            session_id='session-1',
            user_id='user-1',
            anonymous_token_hash=None,
        )

    assert exc_info.value.status_code == 409
    session_repository.commit_lifecycle_changes.assert_not_awaited()


@pytest.mark.asyncio
async def test_create_session_rejects_foreign_existing_session():
    chat_session = ChatSession(
        id=uuid4(),
        session_id='session-1',
        user_id='owner-1',
        service_metadata={},
        last_activity_at=datetime.now(UTC),
    )
    session_repository = AsyncMock(spec=ChatSessionRepository)
    turn_repository = AsyncMock(spec=ChatTurnRepository)
    checkpointer = AsyncMock(spec=AsyncRedisSaver)
    session_repository.lock_or_create_for_lifecycle.return_value = (
        chat_session,
        False,
    )

    with pytest.raises(ChatSessionAccessDeniedError) as exc_info:
        await _service(
            session_repository,
            turn_repository,
            checkpointer,
        ).create_session(
            session_id='session-1',
            user_id='owner-2',
            anonymous_token_hash=None,
        )

    assert exc_info.value.status_code == 403


@pytest.mark.asyncio
async def test_close_session_sets_closed_at_for_owner():
    chat_session = ChatSession(
        id=uuid4(),
        session_id='session-1',
        anonymous_token_hash='a' * 64,
        service_metadata={},
        last_activity_at=datetime.now(UTC),
    )
    session_repository = AsyncMock(spec=ChatSessionRepository)
    turn_repository = AsyncMock(spec=ChatTurnRepository)
    checkpointer = AsyncMock(spec=AsyncRedisSaver)
    session_repository.lock_by_session_id.return_value = chat_session

    result = await _service(
        session_repository,
        turn_repository,
        checkpointer,
    ).close_session(
        session_id='session-1',
        user_id=None,
        anonymous_token_hash='a' * 64,
    )

    assert result.session_id == 'session-1'
    assert result.closed_at == chat_session.closed_at
    assert result.closed_at is not None
    session_repository.commit_lifecycle_changes.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_close_session_keeps_original_closed_at_on_retry():
    original_closed_at = datetime.now(UTC) - timedelta(minutes=1)
    chat_session = ChatSession(
        id=uuid4(),
        session_id='session-1',
        user_id='user-1',
        service_metadata={},
        last_activity_at=datetime.now(UTC),
        closed_at=original_closed_at,
    )
    session_repository = AsyncMock(spec=ChatSessionRepository)
    turn_repository = AsyncMock(spec=ChatTurnRepository)
    checkpointer = AsyncMock(spec=AsyncRedisSaver)
    session_repository.lock_by_session_id.return_value = chat_session

    result = await _service(
        session_repository,
        turn_repository,
        checkpointer,
    ).close_session(
        session_id='session-1',
        user_id='user-1',
        anonymous_token_hash=None,
    )

    assert result.closed_at == original_closed_at
    assert chat_session.closed_at == original_closed_at
    session_repository.commit_lifecycle_changes.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_close_session_raises_not_found_for_missing_session():
    session_repository = AsyncMock(spec=ChatSessionRepository)
    turn_repository = AsyncMock(spec=ChatTurnRepository)
    checkpointer = AsyncMock(spec=AsyncRedisSaver)
    session_repository.lock_by_session_id.return_value = None

    with pytest.raises(ChatSessionNotFoundError) as exc_info:
        await _service(
            session_repository,
            turn_repository,
            checkpointer,
        ).close_session(
            session_id='missing-session',
            user_id='user-1',
            anonymous_token_hash=None,
        )

    assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_close_session_rejects_foreign_owner():
    chat_session = ChatSession(
        id=uuid4(),
        session_id='session-1',
        user_id='owner-1',
        service_metadata={},
        last_activity_at=datetime.now(UTC),
    )
    session_repository = AsyncMock(spec=ChatSessionRepository)
    turn_repository = AsyncMock(spec=ChatTurnRepository)
    checkpointer = AsyncMock(spec=AsyncRedisSaver)
    session_repository.lock_by_session_id.return_value = chat_session

    with pytest.raises(ChatSessionAccessDeniedError) as exc_info:
        await _service(
            session_repository,
            turn_repository,
            checkpointer,
        ).close_session(
            session_id='session-1',
            user_id='owner-2',
            anonymous_token_hash=None,
        )

    assert exc_info.value.status_code == 403
    assert chat_session.closed_at is None
    session_repository.commit_lifecycle_changes.assert_not_awaited()


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
async def test_closed_predecessor_creates_and_retries_exact_successor():
    original_closed_at = datetime.now(UTC) - timedelta(minutes=1)
    predecessor = ChatSession(
        id=uuid4(),
        session_id='session-p',
        user_id='user-1',
        service_metadata={},
        last_activity_at=datetime.now(UTC),
        closed_at=original_closed_at,
    )
    successor = ChatSession(
        id=uuid4(),
        session_id='session-n',
        user_id='user-1',
        service_metadata={},
        last_activity_at=datetime.now(UTC),
    )
    session_repository = AsyncMock(spec=ChatSessionRepository)
    turn_repository = AsyncMock(spec=ChatTurnRepository)
    checkpointer = AsyncMock(spec=AsyncRedisSaver)
    session_repository.lock_or_create_for_lifecycle.side_effect = [
        (predecessor, False),
        (successor, True),
        (predecessor, False),
    ]
    session_repository.lock_by_session_id.return_value = successor
    service = _service(session_repository, turn_repository, checkpointer)

    first = await service.resolve_session(
        session_id='session-p',
        replacement_session_id='session-n',
        user_id='user-1',
        anonymous_token_hash=None,
        refreshed_anonymous_token_hash=None,
        replacement_anonymous_token_hash=None,
    )
    retry = await service.resolve_session(
        session_id='session-p',
        replacement_session_id='session-n',
        user_id='user-1',
        anonymous_token_hash=None,
        refreshed_anonymous_token_hash=None,
        replacement_anonymous_token_hash=None,
    )

    assert first.boundary == retry.boundary == 'expired'
    assert first.session_id == retry.session_id == 'session-n'
    assert predecessor.closed_at == original_closed_at
    assert (
        predecessor.service_metadata[SUCCESSOR_SESSION_ID_METADATA_KEY]
        == 'session-n'
    )
    assert (
        successor.service_metadata[
            SUCCESSOR_RECOVERY_PREDECESSOR_ID_METADATA_KEY
        ]
        == 'session-p'
    )
    assert session_repository.commit_lifecycle_changes.await_count == 2


@pytest.mark.asyncio
async def test_closed_predecessor_reuses_open_owned_successor():
    predecessor = ChatSession(
        id=uuid4(),
        session_id='session-p',
        user_id='user-1',
        service_metadata={},
        last_activity_at=datetime.now(UTC),
        closed_at=datetime.now(UTC),
    )
    previous_successor_activity = datetime.now(UTC) - timedelta(days=2)
    successor = ChatSession(
        id=uuid4(),
        session_id='session-n',
        user_id='user-1',
        service_metadata={},
        last_activity_at=previous_successor_activity,
    )
    session_repository = AsyncMock(spec=ChatSessionRepository)
    turn_repository = AsyncMock(spec=ChatTurnRepository)
    checkpointer = AsyncMock(spec=AsyncRedisSaver)
    session_repository.lock_or_create_for_lifecycle.side_effect = [
        (predecessor, False),
        (successor, False),
    ]

    result = await _service(
        session_repository,
        turn_repository,
        checkpointer,
    ).resolve_session(
        session_id='session-p',
        replacement_session_id='session-n',
        user_id='user-1',
        anonymous_token_hash=None,
        refreshed_anonymous_token_hash=None,
        replacement_anonymous_token_hash=None,
    )

    assert result.boundary == 'expired'
    assert result.session_id == 'session-n'
    assert (
        predecessor.service_metadata[SUCCESSOR_SESSION_ID_METADATA_KEY]
        == 'session-n'
    )
    assert (
        successor.service_metadata[
            SUCCESSOR_RECOVERY_PREDECESSOR_ID_METADATA_KEY
        ]
        == 'session-p'
    )
    assert successor.last_activity_at > previous_successor_activity
    session_repository.commit_lifecycle_changes.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_closed_anonymous_predecessor_reuses_exact_owned_successor():
    predecessor = ChatSession(
        id=uuid4(),
        session_id='session-p',
        anonymous_token_hash='a' * 64,
        service_metadata={},
        last_activity_at=datetime.now(UTC),
        closed_at=datetime.now(UTC),
    )
    successor = ChatSession(
        id=uuid4(),
        session_id='session-n',
        anonymous_token_hash='c' * 64,
        service_metadata={},
        last_activity_at=datetime.now(UTC) - timedelta(days=2),
    )
    session_repository = AsyncMock(spec=ChatSessionRepository)
    turn_repository = AsyncMock(spec=ChatTurnRepository)
    checkpointer = AsyncMock(spec=AsyncRedisSaver)
    session_repository.lock_or_create_for_lifecycle.side_effect = [
        (predecessor, False),
        (successor, False),
    ]

    result = await _service(
        session_repository,
        turn_repository,
        checkpointer,
    ).resolve_session(
        session_id='session-p',
        replacement_session_id='session-n',
        user_id=None,
        anonymous_token_hash='a' * 64,
        refreshed_anonymous_token_hash='b' * 64,
        replacement_anonymous_token_hash='c' * 64,
    )

    assert result.boundary == 'expired'
    assert predecessor.anonymous_token_hash == 'b' * 64
    assert successor.anonymous_token_hash == 'c' * 64
    assert (
        predecessor.service_metadata[SUCCESSOR_SESSION_ID_METADATA_KEY]
        == 'session-n'
    )


@pytest.mark.asyncio
async def test_closed_anonymous_predecessor_rejects_wrong_successor_hash():
    predecessor = ChatSession(
        id=uuid4(),
        session_id='session-p',
        anonymous_token_hash='a' * 64,
        service_metadata={},
        last_activity_at=datetime.now(UTC),
        closed_at=datetime.now(UTC),
    )
    successor = ChatSession(
        id=uuid4(),
        session_id='session-n',
        anonymous_token_hash='c' * 64,
        service_metadata={},
        last_activity_at=datetime.now(UTC),
    )
    session_repository = AsyncMock(spec=ChatSessionRepository)
    turn_repository = AsyncMock(spec=ChatTurnRepository)
    checkpointer = AsyncMock(spec=AsyncRedisSaver)
    session_repository.lock_or_create_for_lifecycle.side_effect = [
        (predecessor, False),
        (successor, False),
    ]

    with pytest.raises(ChatSessionAccessDeniedError):
        await _service(
            session_repository,
            turn_repository,
            checkpointer,
        ).resolve_session(
            session_id='session-p',
            replacement_session_id='session-n',
            user_id=None,
            anonymous_token_hash='a' * 64,
            refreshed_anonymous_token_hash='b' * 64,
            replacement_anonymous_token_hash='d' * 64,
        )

    assert SUCCESSOR_SESSION_ID_METADATA_KEY not in predecessor.service_metadata
    assert successor.anonymous_token_hash == 'c' * 64
    assert successor.service_metadata == {}
    session_repository.commit_lifecycle_changes.assert_not_awaited()


@pytest.mark.asyncio
async def test_open_inactive_predecessor_rejects_existing_owned_successor():
    predecessor = ChatSession(
        id=uuid4(),
        session_id='session-p',
        user_id='user-1',
        service_metadata={},
        last_activity_at=datetime.now(UTC) - timedelta(days=2),
    )
    successor = ChatSession(
        id=uuid4(),
        session_id='session-n',
        user_id='user-1',
        service_metadata={},
        last_activity_at=datetime.now(UTC),
    )
    session_repository = AsyncMock(spec=ChatSessionRepository)
    turn_repository = AsyncMock(spec=ChatTurnRepository)
    checkpointer = AsyncMock(spec=AsyncRedisSaver)
    session_repository.lock_or_create_for_lifecycle.side_effect = [
        (predecessor, False),
        (successor, False),
    ]

    with pytest.raises(ChatSessionResolutionConflictError):
        await _service(
            session_repository,
            turn_repository,
            checkpointer,
        ).resolve_session(
            session_id='session-p',
            replacement_session_id='session-n',
            user_id='user-1',
            anonymous_token_hash=None,
            refreshed_anonymous_token_hash=None,
            replacement_anonymous_token_hash=None,
        )

    assert predecessor.closed_at is None
    assert predecessor.service_metadata == {}
    assert successor.service_metadata == {}
    session_repository.commit_lifecycle_changes.assert_not_awaited()


@pytest.mark.asyncio
async def test_closed_predecessor_rejects_foreign_existing_successor():
    predecessor = ChatSession(
        id=uuid4(),
        session_id='session-p',
        user_id='user-1',
        service_metadata={},
        last_activity_at=datetime.now(UTC),
        closed_at=datetime.now(UTC),
    )
    successor = ChatSession(
        id=uuid4(),
        session_id='session-n',
        user_id='owner-2',
        service_metadata={},
        last_activity_at=datetime.now(UTC),
    )
    session_repository = AsyncMock(spec=ChatSessionRepository)
    turn_repository = AsyncMock(spec=ChatTurnRepository)
    checkpointer = AsyncMock(spec=AsyncRedisSaver)
    session_repository.lock_or_create_for_lifecycle.side_effect = [
        (predecessor, False),
        (successor, False),
    ]

    with pytest.raises(ChatSessionAccessDeniedError):
        await _service(
            session_repository,
            turn_repository,
            checkpointer,
        ).resolve_session(
            session_id='session-p',
            replacement_session_id='session-n',
            user_id='user-1',
            anonymous_token_hash=None,
            refreshed_anonymous_token_hash=None,
            replacement_anonymous_token_hash=None,
        )

    assert predecessor.service_metadata == {}
    assert successor.service_metadata == {}
    session_repository.commit_lifecycle_changes.assert_not_awaited()


@pytest.mark.asyncio
async def test_closed_predecessor_rejects_closed_owned_successor():
    predecessor = ChatSession(
        id=uuid4(),
        session_id='session-p',
        user_id='user-1',
        service_metadata={},
        last_activity_at=datetime.now(UTC),
        closed_at=datetime.now(UTC),
    )
    successor = ChatSession(
        id=uuid4(),
        session_id='session-n',
        user_id='user-1',
        service_metadata={},
        last_activity_at=datetime.now(UTC),
        closed_at=datetime.now(UTC),
    )
    session_repository = AsyncMock(spec=ChatSessionRepository)
    turn_repository = AsyncMock(spec=ChatTurnRepository)
    checkpointer = AsyncMock(spec=AsyncRedisSaver)
    session_repository.lock_or_create_for_lifecycle.side_effect = [
        (predecessor, False),
        (successor, False),
    ]

    with pytest.raises(ChatSessionResolutionConflictError):
        await _service(
            session_repository,
            turn_repository,
            checkpointer,
        ).resolve_session(
            session_id='session-p',
            replacement_session_id='session-n',
            user_id='user-1',
            anonymous_token_hash=None,
            refreshed_anonymous_token_hash=None,
            replacement_anonymous_token_hash=None,
        )

    assert predecessor.service_metadata == {}
    assert successor.service_metadata == {}
    session_repository.commit_lifecycle_changes.assert_not_awaited()


@pytest.mark.asyncio
async def test_closed_predecessor_rejects_successor_bound_to_another():
    predecessor = ChatSession(
        id=uuid4(),
        session_id='session-p',
        user_id='user-1',
        service_metadata={},
        last_activity_at=datetime.now(UTC),
        closed_at=datetime.now(UTC),
    )
    successor = ChatSession(
        id=uuid4(),
        session_id='session-n',
        user_id='user-1',
        service_metadata={
            SUCCESSOR_RECOVERY_PREDECESSOR_ID_METADATA_KEY: 'session-q',
        },
        last_activity_at=datetime.now(UTC),
    )
    original_successor_metadata = dict(successor.service_metadata)
    session_repository = AsyncMock(spec=ChatSessionRepository)
    turn_repository = AsyncMock(spec=ChatTurnRepository)
    checkpointer = AsyncMock(spec=AsyncRedisSaver)
    session_repository.lock_or_create_for_lifecycle.side_effect = [
        (predecessor, False),
        (successor, False),
    ]

    with pytest.raises(ChatSessionResolutionConflictError):
        await _service(
            session_repository,
            turn_repository,
            checkpointer,
        ).resolve_session(
            session_id='session-p',
            replacement_session_id='session-n',
            user_id='user-1',
            anonymous_token_hash=None,
            refreshed_anonymous_token_hash=None,
            replacement_anonymous_token_hash=None,
        )

    assert predecessor.service_metadata == {}
    assert successor.service_metadata == original_successor_metadata
    session_repository.commit_lifecycle_changes.assert_not_awaited()


@pytest.mark.asyncio
async def test_exact_retry_rejects_successor_closed_after_binding():
    predecessor = ChatSession(
        id=uuid4(),
        session_id='session-p',
        user_id='user-1',
        service_metadata={
            SUCCESSOR_SESSION_ID_METADATA_KEY: 'session-n',
        },
        last_activity_at=datetime.now(UTC),
        closed_at=datetime.now(UTC),
    )
    successor = ChatSession(
        id=uuid4(),
        session_id='session-n',
        user_id='user-1',
        service_metadata={
            SUCCESSOR_RECOVERY_PREDECESSOR_ID_METADATA_KEY: 'session-p',
        },
        last_activity_at=datetime.now(UTC),
        closed_at=datetime.now(UTC),
    )
    session_repository = AsyncMock(spec=ChatSessionRepository)
    turn_repository = AsyncMock(spec=ChatTurnRepository)
    checkpointer = AsyncMock(spec=AsyncRedisSaver)
    session_repository.lock_or_create_for_lifecycle.return_value = (
        predecessor,
        False,
    )
    session_repository.lock_by_session_id.return_value = successor

    with pytest.raises(ChatSessionResolutionConflictError):
        await _service(
            session_repository,
            turn_repository,
            checkpointer,
        ).resolve_session(
            session_id='session-p',
            replacement_session_id='session-n',
            user_id='user-1',
            anonymous_token_hash=None,
            refreshed_anonymous_token_hash=None,
            replacement_anonymous_token_hash=None,
        )

    session_repository.commit_lifecycle_changes.assert_not_awaited()


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
