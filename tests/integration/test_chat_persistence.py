import pytest

from app.messaging.schemas import AgentRequestMessage
from app.repositories.chat_session import ChatSessionRepository
from app.repositories.chat_turn import ChatTurnRepository
from app.services.chat_persistence import START_CLAIMED, ChatPersistenceService

pytestmark = pytest.mark.integration


@pytest.mark.parametrize(
    (
        'user_id',
        'anonymous_token_hash',
        'expected_user_id',
        'expected_anonymous_token_hash',
    ),
    [
        ('user-1', None, 'user-1', None),
        (None, 'a' * 64, None, 'a' * 64),
        ('user-1', 'a' * 64, 'user-1', None),
    ],
)
async def test_validated_owner_is_persisted_on_real_postgresql(
    db_session,
    user_id: str | None,
    anonymous_token_hash: str | None,
    expected_user_id: str | None,
    expected_anonymous_token_hash: str | None,
) -> None:
    payload = AgentRequestMessage(
        session_id=f'session-{expected_user_id or "anonymous"}',
        request_id=f'request-{expected_user_id or "anonymous"}',
        user_id=user_id,
        anonymous_token_hash=anonymous_token_hash,
        message='Вопрос',
    )
    session_repository = ChatSessionRepository(db_session)
    service = ChatPersistenceService(
        session_repository,
        ChatTurnRepository(db_session),
    )

    result = await service.start_turn(
        session_id=payload.session_id,
        request_id=payload.request_id,
        user_id=payload.user_id,
        anonymous_token_hash=payload.anonymous_token_hash,
        question=payload.message,
        worker_id='worker-1',
        lease_seconds=900.0,
    )
    saved_session = await session_repository.get_by_session_id(payload.session_id)

    assert result.outcome == START_CLAIMED
    assert saved_session is not None
    assert saved_session.user_id == expected_user_id
    assert saved_session.anonymous_token_hash == expected_anonymous_token_hash
