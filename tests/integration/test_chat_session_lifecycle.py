import asyncio
import uuid
from datetime import UTC, datetime, timedelta

import httpx
import pytest
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.checkpoint.redis_saver import get_redis_checkpointer
from app.core.rate_limit import limiter
from app.core.settings import RedisSettings, get_settings
from app.db.models.chat_session import ChatSession
from app.db.models.chat_turn import (
    STATUS_COMPLETED,
    STATUS_GENERATION_FAILED,
    STATUS_PROCESSING,
    ChatTurn,
)
from app.dependencies.services import get_chat_session_lifecycle_service
from app.exceptions.chat_session import (
    ChatSessionAccessDeniedError,
    ChatSessionResolutionConflictError,
)
from app.graph.state import AgentState
from app.main import app
from app.repositories.chat_session import ChatSessionRepository
from app.repositories.chat_turn import ChatTurnRepository
from app.services.chat_history import ChatHistoryService
from app.services.chat_persistence import START_CLAIMED, ChatPersistenceService
from app.services.chat_session_access import (
    PREVIOUS_ANONYMOUS_HASH_METADATA_KEY,
    ensure_chat_session_access,
)
from app.services.chat_session_lifecycle import (
    ANONYMOUS_RECOVERY_OPERATION_ID_METADATA_KEY,
    SUCCESSOR_RECOVERY_DEADLINE_METADATA_KEY,
    SUCCESSOR_RECOVERY_PREDECESSOR_ID_METADATA_KEY,
    SUCCESSOR_SESSION_ID_METADATA_KEY,
    ChatSessionLifecycleService,
)

pytestmark = pytest.mark.integration

SESSION_TTL_SECONDS = 86400
EXPIRED_PROCESSING_LEASE = 'expired_processing_lease'
MISSING_PROCESSING_LEASE = 'missing_processing_lease'


def _echo_node(state: AgentState) -> dict:
    return {}


def _build_echo_graph():
    builder = StateGraph(AgentState)
    builder.add_node('echo', _echo_node)
    builder.set_entry_point('echo')
    builder.set_finish_point('echo')
    return builder


def _initial_state(session_id: str) -> dict:
    return {
        'session_id': session_id,
        'user_id': None,
        'messages': [HumanMessage(content='Проверка живого контекста')],
        'retrieved_chunks': [],
        'tool_calls': [],
        'search_unavailable': False,
    }


def _redis_settings() -> RedisSettings:
    return RedisSettings(
        redis_host='localhost',
        redis_port=6379,
        redis_session_ttl_seconds=SESSION_TTL_SECONDS,
    )


def _service(db_session, checkpointer) -> ChatSessionLifecycleService:
    return ChatSessionLifecycleService(
        chat_session_repository=ChatSessionRepository(db_session),
        chat_turn_repository=ChatTurnRepository(db_session),
        checkpointer=checkpointer,
        session_ttl_seconds=SESSION_TTL_SECONDS,
    )


async def _save_completed_turn(
    db_session,
    chat_session: ChatSession,
    *,
    request_id: str,
) -> None:
    await ChatTurnRepository(db_session).save(
        ChatTurn(
            request_id=request_id,
            chat_session_id=chat_session.id,
            sequence_number=1,
            user_id=chat_session.user_id,
            question='Вопрос',
            answer='Ответ',
            status=STATUS_COMPLETED,
        )
    )


async def test_missing_anonymous_session_is_created_with_current_owner(
    db_session,
):
    current_id = str(uuid.uuid4())

    async with get_redis_checkpointer(_redis_settings()) as checkpointer:
        result = await _service(db_session, checkpointer).resolve_session(
            session_id=current_id,
            replacement_session_id=str(uuid.uuid4()),
            user_id=None,
            anonymous_token_hash='a' * 64,
            refreshed_anonymous_token_hash='b' * 64,
            replacement_anonymous_token_hash='c' * 64,
        )

    current = await ChatSessionRepository(db_session).get_by_session_id(
        current_id
    )
    assert result.boundary == 'created'
    assert result.session_id == current_id
    assert result.previous_session_id is None
    assert current is not None
    assert current.anonymous_token_hash == 'a' * 64


async def test_explicit_auth_close_is_idempotent_and_excluded_from_current(
    db_engine,
    db_session,
):
    session_id = str(uuid.uuid4())
    session_factory = async_sessionmaker(db_engine, expire_on_commit=False)

    async with get_redis_checkpointer(_redis_settings()) as checkpointer:
        async def lifecycle_service_override():
            async with session_factory() as request_session:
                yield _service(request_session, checkpointer)

        app.dependency_overrides[
            get_chat_session_lifecycle_service
        ] = lifecycle_service_override
        limiter.reset()
        try:
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport,
                base_url='http://test',
            ) as client:
                headers = {
                    'X-API-Key': (
                        get_settings().app.api_key.get_secret_value()
                    ),
                    'X-Vera-User-ID': 'user-1',
                }
                payload = {'session_id': session_id}
                created = await client.post(
                    '/api/v1/chat/sessions',
                    headers=headers,
                    json=payload,
                )
                create_retry = await client.post(
                    '/api/v1/chat/sessions',
                    headers=headers,
                    json=payload,
                )
                current_before = await client.get(
                    '/api/v1/chat/sessions/current',
                    headers=headers,
                )
                closed = await client.post(
                    f'/api/v1/chat/sessions/{session_id}/close',
                    headers=headers,
                )
                close_retry = await client.post(
                    f'/api/v1/chat/sessions/{session_id}/close',
                    headers=headers,
                )
                closed_create = await client.post(
                    '/api/v1/chat/sessions',
                    headers=headers,
                    json=payload,
                )
                current_after = await client.get(
                    '/api/v1/chat/sessions/current',
                    headers=headers,
                )
        finally:
            app.dependency_overrides.clear()
            limiter.reset()

    persisted = await ChatSessionRepository(db_session).get_by_session_id(
        session_id
    )
    row_count = (
        await db_session.execute(
            select(func.count(ChatSession.id)).where(
                ChatSession.session_id == session_id
            )
        )
    ).scalar_one()

    expected_created = {
        'session_id': session_id,
        'session_ttl_seconds': SESSION_TTL_SECONDS,
    }
    assert created.status_code == 200
    assert create_retry.status_code == 200
    assert created.json() == create_retry.json() == expected_created
    assert current_before.json() == {'session_id': session_id}
    assert closed.status_code == 200
    assert close_retry.status_code == 200
    assert closed.json() == close_retry.json()
    assert closed_create.status_code == 409
    assert current_after.json() == {'session_id': None}
    assert row_count == 1
    assert persisted is not None
    assert persisted.closed_at is not None


async def test_explicit_create_and_close_reject_foreign_owner(
    db_engine,
    db_session,
):
    session_id = str(uuid.uuid4())
    persisted = await ChatSessionRepository(db_session).save(
        ChatSession(
            session_id=session_id,
            user_id='owner-1',
            last_activity_at=datetime.now(UTC),
        )
    )
    session_factory = async_sessionmaker(db_engine, expire_on_commit=False)

    async with get_redis_checkpointer(_redis_settings()) as checkpointer:
        async def lifecycle_service_override():
            async with session_factory() as request_session:
                yield _service(request_session, checkpointer)

        app.dependency_overrides[
            get_chat_session_lifecycle_service
        ] = lifecycle_service_override
        limiter.reset()
        try:
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport,
                base_url='http://test',
            ) as client:
                headers = {
                    'X-API-Key': (
                        get_settings().app.api_key.get_secret_value()
                    ),
                    'X-Vera-User-ID': 'owner-2',
                }
                create_response = await client.post(
                    '/api/v1/chat/sessions',
                    headers=headers,
                    json={'session_id': session_id},
                )
                close_response = await client.post(
                    f'/api/v1/chat/sessions/{session_id}/close',
                    headers=headers,
                )
        finally:
            app.dependency_overrides.clear()
            limiter.reset()

    await db_session.refresh(persisted)
    assert create_response.status_code == 403
    assert close_response.status_code == 403
    assert persisted.closed_at is None


async def test_anonymous_session_is_assigned_on_authenticated_start_turn(
    db_engine,
    db_session,
):
    session_id = str(uuid.uuid4())
    request_id = str(uuid.uuid4())
    anonymous_hash = 'a' * 64
    session_factory = async_sessionmaker(db_engine, expire_on_commit=False)

    async with get_redis_checkpointer(_redis_settings()) as checkpointer:
        async def lifecycle_service_override():
            async with session_factory() as request_session:
                yield _service(request_session, checkpointer)

        app.dependency_overrides[
            get_chat_session_lifecycle_service
        ] = lifecycle_service_override
        try:
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport,
                base_url='http://test',
            ) as client:
                created = await client.post(
                    '/api/v1/chat/sessions',
                    headers={
                        'X-API-Key': (
                            get_settings().app.api_key.get_secret_value()
                        ),
                        'X-Vera-Anonymous-Token-Hash': anonymous_hash,
                    },
                    json={'session_id': session_id},
                )
        finally:
            app.dependency_overrides.clear()

        anonymous_session = await ChatSessionRepository(
            db_session
        ).get_by_session_id(session_id)
        assert anonymous_session is not None
        await _save_completed_turn(
            db_session,
            anonymous_session,
            request_id=str(uuid.uuid4()),
        )
        graph = _build_echo_graph().compile(checkpointer=checkpointer)
        await graph.ainvoke(
            _initial_state(session_id),
            config={'configurable': {'thread_id': session_id}},
        )
        checkpoint = await checkpointer.aget_tuple(
            {'configurable': {'thread_id': session_id}}
        )
        assert checkpoint is not None

        start_result = await ChatPersistenceService(
            ChatSessionRepository(db_session),
            ChatTurnRepository(db_session),
            checkpointer=checkpointer,
            session_ttl_seconds=SESSION_TTL_SECONDS,
        ).start_turn(
            session_id=session_id,
            request_id=request_id,
            user_id='user-after-login',
            anonymous_token_hash=anonymous_hash,
            question='Продолжить текущий диалог',
            worker_id='worker-1',
            lease_seconds=900.0,
        )

    persisted = await ChatSessionRepository(db_session).get_by_session_id(
        session_id
    )
    row_count = (
        await db_session.execute(select(func.count(ChatSession.id)))
    ).scalar_one()
    turn_count = (
        await db_session.execute(select(func.count(ChatTurn.id)))
    ).scalar_one()
    assert created.status_code == 200
    assert start_result.outcome == START_CLAIMED
    assert row_count == 1
    assert turn_count == 2
    assert persisted is not None
    assert persisted.session_id == session_id
    assert persisted.user_id == 'user-after-login'
    assert persisted.anonymous_token_hash is None


@pytest.mark.parametrize(
    'stale_source',
    [
        pytest.param('database-ttl', id='database-ttl'),
        pytest.param('missing-checkpoint', id='missing-checkpoint'),
    ],
)
async def test_auth_discovery_keeps_predecessor_until_resolve_via_real_api(
    db_engine,
    db_session,
    stale_source: str,
):
    current_id = str(uuid.uuid4())
    replacement_id = str(uuid.uuid4())
    original_last_activity_at = (
        datetime.now(UTC) - timedelta(days=2)
        if stale_source == 'database-ttl'
        else datetime.now(UTC) - timedelta(minutes=1)
    )
    current = await ChatSessionRepository(db_session).save(
        ChatSession(
            session_id=current_id,
            user_id='user-1',
            last_activity_at=original_last_activity_at,
        )
    )
    if stale_source == 'missing-checkpoint':
        await _save_completed_turn(
            db_session,
            current,
            request_id=str(uuid.uuid4()),
        )

    session_factory = async_sessionmaker(db_engine, expire_on_commit=False)

    async with get_redis_checkpointer(_redis_settings()) as checkpointer:
        async def lifecycle_service_override():
            async with session_factory() as request_session:
                yield _service(request_session, checkpointer)

        app.dependency_overrides[
            get_chat_session_lifecycle_service
        ] = lifecycle_service_override
        limiter.reset()
        try:
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport,
                base_url='http://test',
            ) as client:
                discovery_response = await client.get(
                    '/api/v1/chat/sessions/current',
                    headers={
                        'X-API-Key': (
                            get_settings().app.api_key.get_secret_value()
                        ),
                        'X-Vera-User-ID': 'user-1',
                    },
                )
                retry_discovery_response = await client.get(
                    '/api/v1/chat/sessions/current',
                    headers={
                        'X-API-Key': (
                            get_settings().app.api_key.get_secret_value()
                        ),
                        'X-Vera-User-ID': 'user-1',
                    },
                )

                await db_session.refresh(current)
                assert discovery_response.status_code == 200
                assert retry_discovery_response.status_code == 200
                assert discovery_response.json() == {
                    'session_id': current_id
                }
                assert retry_discovery_response.json() == {
                    'session_id': current_id
                }
                assert current.closed_at is None
                assert current.last_activity_at == original_last_activity_at

                response = await client.post(
                    '/api/v1/chat/sessions/resolve',
                    headers={
                        'X-API-Key': (
                            get_settings().app.api_key.get_secret_value()
                        ),
                        'X-Vera-User-ID': 'user-1',
                    },
                    json={
                        'session_id': current_id,
                        'replacement_session_id': replacement_id,
                    },
                )
        finally:
            app.dependency_overrides.clear()
            limiter.reset()

    await db_session.refresh(current)
    replacement = await ChatSessionRepository(db_session).get_by_session_id(
        replacement_id
    )
    assert response.status_code == 200
    assert response.json() == {
        'session_id': replacement_id,
        'previous_session_id': current_id,
        'boundary': 'expired',
        'session_ttl_seconds': SESSION_TTL_SECONDS,
    }
    assert current.closed_at is not None
    assert replacement is not None
    assert replacement.user_id == 'user-1'


async def test_old_created_fresh_anonymous_session_with_checkpoint_is_retained(
    db_session,
):
    current_id = str(uuid.uuid4())
    replacement_id = str(uuid.uuid4())
    current = await ChatSessionRepository(db_session).save(
        ChatSession(
            session_id=current_id,
            anonymous_token_hash='a' * 64,
            created_at=datetime.now(UTC) - timedelta(days=2),
            last_activity_at=datetime.now(UTC) - timedelta(minutes=1),
        )
    )
    await _save_completed_turn(
        db_session,
        current,
        request_id=str(uuid.uuid4()),
    )

    async with get_redis_checkpointer(_redis_settings()) as checkpointer:
        graph = _build_echo_graph().compile(checkpointer=checkpointer)
        await graph.ainvoke(
            _initial_state(current_id),
            config={'configurable': {'thread_id': current_id}},
        )

        result = await _service(db_session, checkpointer).resolve_session(
            session_id=current_id,
            replacement_session_id=replacement_id,
            user_id=None,
            anonymous_token_hash='a' * 64,
            refreshed_anonymous_token_hash='b' * 64,
            replacement_anonymous_token_hash='c' * 64,
        )
        with pytest.raises(ChatSessionAccessDeniedError):
            await _service(db_session, checkpointer).resolve_session(
                session_id=current_id,
                replacement_session_id=replacement_id,
                user_id=None,
                anonymous_token_hash='a' * 64,
                refreshed_anonymous_token_hash='d' * 64,
                replacement_anonymous_token_hash='c' * 64,
            )
        await db_session.rollback()
        retry_result = await _service(
            db_session,
            checkpointer,
        ).resolve_session(
            session_id=current_id,
            replacement_session_id=replacement_id,
            user_id=None,
            anonymous_token_hash='a' * 64,
            refreshed_anonymous_token_hash='b' * 64,
            replacement_anonymous_token_hash='c' * 64,
        )
        checkpoint = await checkpointer.aget_tuple(
            {'configurable': {'thread_id': current_id}}
        )

    await db_session.refresh(current)
    assert result.boundary == 'retained'
    assert retry_result.boundary == 'retained'
    assert result.session_id == current_id
    assert current.closed_at is None
    assert current.anonymous_token_hash == 'b' * 64
    assert (
        current.service_metadata[PREVIOUS_ANONYMOUS_HASH_METADATA_KEY]
        == 'a' * 64
    )
    assert (
        current.service_metadata[
            ANONYMOUS_RECOVERY_OPERATION_ID_METADATA_KEY
        ]
        == replacement_id
    )
    assert current.created_at < datetime.now(UTC) - timedelta(days=1)
    assert current.last_activity_at > datetime.now(UTC) - timedelta(minutes=1)
    assert checkpoint is not None
    history_service = ChatHistoryService(
        ChatSessionRepository(db_session),
        ChatTurnRepository(db_session),
    )
    page = await history_service.get_history(
        current_id,
        user_id=None,
        anonymous_token_hash='a' * 64,
        limit=30,
        before_sequence=None,
    )
    assert len(page.turns) == 1
    await history_service.get_history(
        current_id,
        user_id=None,
        anonymous_token_hash='b' * 64,
        limit=30,
        before_sequence=None,
    )
    with pytest.raises(ChatSessionAccessDeniedError):
        await history_service.get_history(
            current_id,
            user_id=None,
            anonymous_token_hash='d' * 64,
            limit=30,
            before_sequence=None,
        )


async def test_session_with_turns_and_missing_checkpoint_is_replaced(
    db_session,
):
    current_id = str(uuid.uuid4())
    replacement_id = str(uuid.uuid4())
    current = await ChatSessionRepository(db_session).save(
        ChatSession(
            session_id=current_id,
            user_id='user-1',
            last_activity_at=datetime.now(UTC),
        )
    )
    await _save_completed_turn(
        db_session,
        current,
        request_id=str(uuid.uuid4()),
    )

    async with get_redis_checkpointer(_redis_settings()) as checkpointer:
        result = await _service(db_session, checkpointer).resolve_session(
            session_id=current_id,
            replacement_session_id=replacement_id,
            user_id='user-1',
            anonymous_token_hash=None,
            refreshed_anonymous_token_hash=None,
            replacement_anonymous_token_hash=None,
        )

    await db_session.refresh(current)
    replacement = await ChatSessionRepository(db_session).get_by_session_id(
        replacement_id
    )
    assert result.boundary == 'expired'
    assert result.session_id == replacement_id
    assert current.closed_at is not None
    assert replacement is not None


@pytest.mark.parametrize(
    'context_end',
    [
        pytest.param(STATUS_COMPLETED, id='completed'),
        pytest.param(STATUS_GENERATION_FAILED, id='generation-failed'),
        pytest.param(
            EXPIRED_PROCESSING_LEASE,
            id='expired-processing-lease',
        ),
        pytest.param(
            MISSING_PROCESSING_LEASE,
            id='missing-processing-lease',
        ),
    ],
)
async def test_processing_window_retains_only_while_lease_is_live(
    db_engine,
    db_session,
    context_end: str,
):
    current_id = str(uuid.uuid4())
    replacement_id = str(uuid.uuid4())
    request_id = str(uuid.uuid4())
    session_factory = async_sessionmaker(db_engine, expire_on_commit=False)

    async with get_redis_checkpointer(_redis_settings()) as checkpointer:
        start_result = await ChatPersistenceService(
            ChatSessionRepository(db_session),
            ChatTurnRepository(db_session),
            checkpointer=checkpointer,
            session_ttl_seconds=SESSION_TTL_SECONDS,
        ).start_turn(
            session_id=current_id,
            request_id=request_id,
            user_id='processing-user',
            anonymous_token_hash=None,
            question='Вопрос в обработке',
            worker_id='worker-1',
            lease_seconds=900.0,
        )

        checkpoint_config = {
            'configurable': {'thread_id': current_id}
        }
        assert start_result.outcome == START_CLAIMED
        assert await checkpointer.aget_tuple(checkpoint_config) is None

        # Отдельная request-транзакция приходит, пока graph ещё не успел
        # сохранить первый checkpoint для durable processing-реплики.
        async with session_factory() as resolve_session:
            retained = await _service(
                resolve_session,
                checkpointer,
            ).resolve_session(
                session_id=current_id,
                replacement_session_id=replacement_id,
                user_id='processing-user',
                anonymous_token_hash=None,
                refreshed_anonymous_token_hash=None,
                replacement_anonymous_token_hash=None,
            )

        assert retained.boundary == 'retained'
        assert retained.session_id == current_id
        assert await checkpointer.aget_tuple(checkpoint_config) is None

        async with session_factory() as terminal_session:
            turn_repository = ChatTurnRepository(terminal_session)
            turn = await turn_repository.get_by_request_id(request_id)
            assert turn is not None
            if context_end == STATUS_COMPLETED:
                await turn_repository.complete(
                    chat_turn=turn,
                    answer='Ответ',
                    sources=[],
                    technical_metadata={},
                    latency_ms=1,
                )
            elif context_end == STATUS_GENERATION_FAILED:
                await turn_repository.fail(
                    chat_turn=turn,
                    status=STATUS_GENERATION_FAILED,
                    safe_error='GenerationFailed',
                    answer=None,
                    latency_ms=1,
                )
            else:
                assert turn.status == STATUS_PROCESSING
                turn.lease_until = (
                    datetime.now(UTC) - timedelta(seconds=1)
                    if context_end == EXPIRED_PROCESSING_LEASE
                    else None
                )
                await terminal_session.commit()

        async with session_factory() as final_resolve_session:
            expired = await _service(
                final_resolve_session,
                checkpointer,
            ).resolve_session(
                session_id=current_id,
                replacement_session_id=replacement_id,
                user_id='processing-user',
                anonymous_token_hash=None,
                refreshed_anonymous_token_hash=None,
                replacement_anonymous_token_hash=None,
            )

    async with session_factory() as verification_session:
        current = await ChatSessionRepository(
            verification_session
        ).get_by_session_id(current_id)
        replacement = await ChatSessionRepository(
            verification_session
        ).get_by_session_id(replacement_id)

    assert expired.boundary == 'expired'
    assert expired.session_id == replacement_id
    assert expired.previous_session_id == current_id
    assert current is not None
    assert current.closed_at is not None
    assert replacement is not None


async def test_expired_retry_requires_same_real_anonymous_hashes(
    db_session,
):
    current_id = str(uuid.uuid4())
    replacement_id = str(uuid.uuid4())
    old_hash = 'a' * 64
    first_replacement_hash = 'b' * 64
    different_replacement_hash = 'c' * 64
    refreshed_predecessor_hash = 'd' * 64
    different_refreshed_predecessor_hash = 'e' * 64
    normal_successor_hash = 'f' * 64
    await ChatSessionRepository(db_session).save(
        ChatSession(
            session_id=current_id,
            anonymous_token_hash=old_hash,
            last_activity_at=datetime.now(UTC) - timedelta(days=2),
        )
    )

    async with get_redis_checkpointer(_redis_settings()) as checkpointer:
        service = _service(db_session, checkpointer)
        first = await service.resolve_session(
            session_id=current_id,
            replacement_session_id=replacement_id,
            user_id=None,
            anonymous_token_hash=old_hash,
            refreshed_anonymous_token_hash=(
                refreshed_predecessor_hash
            ),
            replacement_anonymous_token_hash=first_replacement_hash,
        )
        with pytest.raises(ChatSessionAccessDeniedError):
            await service.resolve_session(
                session_id=current_id,
                replacement_session_id=replacement_id,
                user_id=None,
                anonymous_token_hash=old_hash,
                refreshed_anonymous_token_hash=(
                    different_refreshed_predecessor_hash
                ),
                replacement_anonymous_token_hash=first_replacement_hash,
            )
        await db_session.rollback()
        with pytest.raises(ChatSessionAccessDeniedError):
            await service.resolve_session(
                session_id=current_id,
                replacement_session_id=replacement_id,
                user_id=None,
                anonymous_token_hash=old_hash,
                refreshed_anonymous_token_hash=(
                    refreshed_predecessor_hash
                ),
                replacement_anonymous_token_hash=(
                    different_replacement_hash
                ),
            )
        await db_session.rollback()
        retry = await service.resolve_session(
            session_id=current_id,
            replacement_session_id=replacement_id,
            user_id=None,
            anonymous_token_hash=old_hash,
            refreshed_anonymous_token_hash=(
                refreshed_predecessor_hash
            ),
            replacement_anonymous_token_hash=first_replacement_hash,
        )
        successor = await ChatSessionRepository(
            db_session
        ).get_by_session_id(replacement_id)
        assert successor is not None
        ensure_chat_session_access(
            successor,
            user_id=None,
            anonymous_token_hash=first_replacement_hash,
        )

        normal = await service.resolve_session(
            session_id=replacement_id,
            replacement_session_id=str(uuid.uuid4()),
            user_id=None,
            anonymous_token_hash=first_replacement_hash,
            refreshed_anonymous_token_hash=normal_successor_hash,
            replacement_anonymous_token_hash='1' * 64,
        )
        with pytest.raises(ChatSessionAccessDeniedError):
            await service.resolve_session(
                session_id=current_id,
                replacement_session_id=replacement_id,
                user_id=None,
                anonymous_token_hash=refreshed_predecessor_hash,
                refreshed_anonymous_token_hash='2' * 64,
                replacement_anonymous_token_hash='3' * 64,
            )
        await db_session.rollback()

    predecessor = await ChatSessionRepository(db_session).get_by_session_id(
        current_id
    )
    successor = await ChatSessionRepository(db_session).get_by_session_id(
        replacement_id
    )
    row_count = (
        await db_session.execute(select(func.count(ChatSession.id)))
    ).scalar_one()
    assert first.boundary == retry.boundary == 'expired'
    assert first.session_id == retry.session_id == replacement_id
    assert normal.boundary == 'retained'
    assert row_count == 2
    assert predecessor is not None
    assert successor is not None
    assert predecessor.anonymous_token_hash == (
        refreshed_predecessor_hash
    )
    assert (
        predecessor.service_metadata[PREVIOUS_ANONYMOUS_HASH_METADATA_KEY]
        == old_hash
    )
    assert successor.anonymous_token_hash == normal_successor_hash
    assert (
        SUCCESSOR_RECOVERY_PREDECESSOR_ID_METADATA_KEY
        not in successor.service_metadata
    )
    assert (
        SUCCESSOR_RECOVERY_DEADLINE_METADATA_KEY
        not in successor.service_metadata
    )
    history_service = ChatHistoryService(
        ChatSessionRepository(db_session),
        ChatTurnRepository(db_session),
    )
    await history_service.get_history(
        current_id,
        user_id=None,
        anonymous_token_hash=refreshed_predecessor_hash,
        limit=30,
        before_sequence=None,
    )
    await history_service.get_history(
        current_id,
        user_id=None,
        anonymous_token_hash=old_hash,
        limit=30,
        before_sequence=None,
    )
    with pytest.raises(ChatSessionAccessDeniedError):
        await history_service.get_history(
            current_id,
            user_id=None,
            anonymous_token_hash=different_refreshed_predecessor_hash,
            limit=30,
            before_sequence=None,
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
            anonymous_token_hash=different_replacement_hash,
        )


async def test_closed_predecessor_creates_and_exactly_retries_successor(
    db_session,
):
    predecessor_id = str(uuid.uuid4())
    successor_id = str(uuid.uuid4())
    original_closed_at = datetime.now(UTC) - timedelta(minutes=1)
    await ChatSessionRepository(db_session).save(
        ChatSession(
            session_id=predecessor_id,
            user_id='user-1',
            last_activity_at=datetime.now(UTC),
            closed_at=original_closed_at,
        )
    )

    async with get_redis_checkpointer(_redis_settings()) as checkpointer:
        service = _service(db_session, checkpointer)
        first = await service.resolve_session(
            session_id=predecessor_id,
            replacement_session_id=successor_id,
            user_id='user-1',
            anonymous_token_hash=None,
            refreshed_anonymous_token_hash=None,
            replacement_anonymous_token_hash=None,
        )
        retry = await service.resolve_session(
            session_id=predecessor_id,
            replacement_session_id=successor_id,
            user_id='user-1',
            anonymous_token_hash=None,
            refreshed_anonymous_token_hash=None,
            replacement_anonymous_token_hash=None,
        )

    predecessor = await ChatSessionRepository(
        db_session
    ).get_by_session_id(predecessor_id)
    successor = await ChatSessionRepository(db_session).get_by_session_id(
        successor_id
    )
    row_count = (
        await db_session.execute(select(func.count(ChatSession.id)))
    ).scalar_one()

    assert first.boundary == retry.boundary == 'expired'
    assert first.session_id == retry.session_id == successor_id
    assert row_count == 2
    assert predecessor is not None
    assert predecessor.closed_at == original_closed_at
    assert (
        predecessor.service_metadata[SUCCESSOR_SESSION_ID_METADATA_KEY]
        == successor_id
    )
    assert successor is not None
    assert successor.closed_at is None


async def test_closed_predecessor_reuses_existing_open_owned_successor(
    db_session,
):
    predecessor_id = str(uuid.uuid4())
    successor_id = str(uuid.uuid4())
    previous_successor_activity = datetime.now(UTC) - timedelta(days=2)
    await ChatSessionRepository(db_session).save(
        ChatSession(
            session_id=predecessor_id,
            anonymous_token_hash='a' * 64,
            last_activity_at=datetime.now(UTC),
            closed_at=datetime.now(UTC),
        )
    )
    await ChatSessionRepository(db_session).save(
        ChatSession(
            session_id=successor_id,
            anonymous_token_hash='c' * 64,
            last_activity_at=previous_successor_activity,
        )
    )

    async with get_redis_checkpointer(_redis_settings()) as checkpointer:
        result = await _service(
            db_session,
            checkpointer,
        ).resolve_session(
            session_id=predecessor_id,
            replacement_session_id=successor_id,
            user_id=None,
            anonymous_token_hash='a' * 64,
            refreshed_anonymous_token_hash='b' * 64,
            replacement_anonymous_token_hash='c' * 64,
        )

    predecessor = await ChatSessionRepository(
        db_session
    ).get_by_session_id(predecessor_id)
    successor = await ChatSessionRepository(db_session).get_by_session_id(
        successor_id
    )
    row_count = (
        await db_session.execute(select(func.count(ChatSession.id)))
    ).scalar_one()

    assert result.boundary == 'expired'
    assert result.session_id == successor_id
    assert row_count == 2
    assert predecessor is not None
    assert (
        predecessor.service_metadata[SUCCESSOR_SESSION_ID_METADATA_KEY]
        == successor_id
    )
    assert predecessor.anonymous_token_hash == 'b' * 64
    assert successor is not None
    assert successor.anonymous_token_hash == 'c' * 64
    assert successor.last_activity_at > previous_successor_activity
    assert (
        successor.service_metadata[
            SUCCESSOR_RECOVERY_PREDECESSOR_ID_METADATA_KEY
        ]
        == predecessor_id
    )


async def test_closed_predecessor_rejects_wrong_successor_hash(
    db_session,
):
    predecessor_id = str(uuid.uuid4())
    successor_id = str(uuid.uuid4())
    predecessor_closed_at = datetime.now(UTC) - timedelta(minutes=1)
    await ChatSessionRepository(db_session).save(
        ChatSession(
            session_id=predecessor_id,
            anonymous_token_hash='a' * 64,
            service_metadata={},
            last_activity_at=datetime.now(UTC),
            closed_at=predecessor_closed_at,
        )
    )
    await ChatSessionRepository(db_session).save(
        ChatSession(
            session_id=successor_id,
            anonymous_token_hash='c' * 64,
            service_metadata={},
            last_activity_at=datetime.now(UTC),
        )
    )

    async with get_redis_checkpointer(_redis_settings()) as checkpointer:
        with pytest.raises(ChatSessionAccessDeniedError):
            await _service(db_session, checkpointer).resolve_session(
                session_id=predecessor_id,
                replacement_session_id=successor_id,
                user_id=None,
                anonymous_token_hash='a' * 64,
                refreshed_anonymous_token_hash='b' * 64,
                replacement_anonymous_token_hash='d' * 64,
            )
        await db_session.rollback()

    predecessor = await ChatSessionRepository(
        db_session
    ).get_by_session_id(predecessor_id)
    successor = await ChatSessionRepository(db_session).get_by_session_id(
        successor_id
    )
    row_count = (
        await db_session.execute(select(func.count(ChatSession.id)))
    ).scalar_one()

    assert row_count == 2
    assert predecessor is not None
    assert predecessor.closed_at == predecessor_closed_at
    assert predecessor.anonymous_token_hash == 'a' * 64
    assert SUCCESSOR_SESSION_ID_METADATA_KEY not in predecessor.service_metadata
    assert successor is not None
    assert successor.closed_at is None
    assert successor.anonymous_token_hash == 'c' * 64
    assert successor.service_metadata == {}


@pytest.mark.parametrize(
    (
        'successor_owner',
        'successor_closed',
        'bound_predecessor_id',
        'expected_error',
    ),
    [
        pytest.param(
            'owner-2',
            False,
            None,
            ChatSessionAccessDeniedError,
            id='foreign',
        ),
        pytest.param(
            'user-1',
            True,
            None,
            ChatSessionResolutionConflictError,
            id='closed-owned',
        ),
        pytest.param(
            'user-1',
            False,
            'session-q',
            ChatSessionResolutionConflictError,
            id='bound-to-another',
        ),
    ],
)
async def test_closed_predecessor_rejects_unsafe_existing_successor(
    db_session,
    successor_owner: str,
    successor_closed: bool,
    bound_predecessor_id: str | None,
    expected_error: type[Exception],
):
    predecessor_id = str(uuid.uuid4())
    successor_id = str(uuid.uuid4())
    predecessor_closed_at = datetime.now(UTC) - timedelta(minutes=1)
    successor_closed_at = (
        datetime.now(UTC) if successor_closed else None
    )
    await ChatSessionRepository(db_session).save(
        ChatSession(
            session_id=predecessor_id,
            user_id='user-1',
            service_metadata={},
            last_activity_at=datetime.now(UTC),
            closed_at=predecessor_closed_at,
        )
    )
    successor_metadata = (
        {
            SUCCESSOR_RECOVERY_PREDECESSOR_ID_METADATA_KEY: (
                bound_predecessor_id
            )
        }
        if bound_predecessor_id is not None
        else {}
    )
    await ChatSessionRepository(db_session).save(
        ChatSession(
            session_id=successor_id,
            user_id=successor_owner,
            service_metadata=successor_metadata,
            last_activity_at=datetime.now(UTC),
            closed_at=successor_closed_at,
        )
    )

    async with get_redis_checkpointer(_redis_settings()) as checkpointer:
        with pytest.raises(expected_error):
            await _service(db_session, checkpointer).resolve_session(
                session_id=predecessor_id,
                replacement_session_id=successor_id,
                user_id='user-1',
                anonymous_token_hash=None,
                refreshed_anonymous_token_hash=None,
                replacement_anonymous_token_hash=None,
            )
        await db_session.rollback()

    predecessor = await ChatSessionRepository(
        db_session
    ).get_by_session_id(predecessor_id)
    successor = await ChatSessionRepository(db_session).get_by_session_id(
        successor_id
    )
    row_count = (
        await db_session.execute(select(func.count(ChatSession.id)))
    ).scalar_one()

    assert row_count == 2
    assert predecessor is not None
    assert predecessor.closed_at == predecessor_closed_at
    assert SUCCESSOR_SESSION_ID_METADATA_KEY not in predecessor.service_metadata
    assert successor is not None
    assert successor.closed_at == successor_closed_at
    assert successor.service_metadata == successor_metadata


async def test_concurrent_closed_predecessor_retries_keep_one_successor(
    db_engine,
    db_session,
):
    predecessor_id = str(uuid.uuid4())
    successor_id = str(uuid.uuid4())
    await ChatSessionRepository(db_session).save(
        ChatSession(
            session_id=predecessor_id,
            anonymous_token_hash='a' * 64,
            last_activity_at=datetime.now(UTC),
            closed_at=datetime.now(UTC),
        )
    )
    session_factory = async_sessionmaker(db_engine, expire_on_commit=False)

    async with get_redis_checkpointer(_redis_settings()) as checkpointer:
        async def resolve_once():
            async with session_factory() as request_session:
                return await _service(
                    request_session,
                    checkpointer,
                ).resolve_session(
                    session_id=predecessor_id,
                    replacement_session_id=successor_id,
                    user_id=None,
                    anonymous_token_hash='a' * 64,
                    refreshed_anonymous_token_hash='b' * 64,
                    replacement_anonymous_token_hash='c' * 64,
                )

        first, second = await asyncio.gather(
            resolve_once(),
            resolve_once(),
        )

    async with session_factory() as verification_session:
        row_count = (
            await verification_session.execute(
                select(func.count(ChatSession.id))
            )
        ).scalar_one()
        predecessor = await ChatSessionRepository(
            verification_session
        ).get_by_session_id(predecessor_id)
        successor = await ChatSessionRepository(
            verification_session
        ).get_by_session_id(successor_id)

    assert first.boundary == second.boundary == 'expired'
    assert first.session_id == second.session_id == successor_id
    assert row_count == 2
    assert predecessor is not None
    assert (
        predecessor.service_metadata[SUCCESSOR_SESSION_ID_METADATA_KEY]
        == successor_id
    )
    assert predecessor.anonymous_token_hash == 'b' * 64
    assert successor is not None
    assert successor.anonymous_token_hash == 'c' * 64


async def test_concurrent_exact_expiry_retries_keep_one_successor_and_hashes(
    db_engine,
):
    current_id = str(uuid.uuid4())
    replacement_id = str(uuid.uuid4())
    session_factory = async_sessionmaker(db_engine, expire_on_commit=False)
    async with session_factory() as setup_session:
        await ChatSessionRepository(setup_session).save(
            ChatSession(
                session_id=current_id,
                anonymous_token_hash='a' * 64,
                last_activity_at=datetime.now(UTC) - timedelta(days=2),
            )
        )

    async with get_redis_checkpointer(_redis_settings()) as checkpointer:

        async def resolve_once():
            async with session_factory() as request_session:
                return await _service(
                    request_session,
                    checkpointer,
                ).resolve_session(
                    session_id=current_id,
                    replacement_session_id=replacement_id,
                    user_id=None,
                    anonymous_token_hash='a' * 64,
                    refreshed_anonymous_token_hash='b' * 64,
                    replacement_anonymous_token_hash='c' * 64,
                )

        first, second = await asyncio.gather(
            resolve_once(),
            resolve_once(),
        )

    async with session_factory() as verification_session:
        row_count = (
            await verification_session.execute(
                select(func.count(ChatSession.id))
            )
        ).scalar_one()
        current = await ChatSessionRepository(
            verification_session
        ).get_by_session_id(current_id)
        successor = await ChatSessionRepository(
            verification_session
        ).get_by_session_id(replacement_id)

    assert first.session_id == second.session_id == replacement_id
    assert first.boundary == second.boundary == 'expired'
    assert row_count == 2
    assert current is not None
    assert current.closed_at is not None
    assert current.anonymous_token_hash == 'b' * 64
    assert successor is not None
    assert successor.anonymous_token_hash == 'c' * 64
