import pytest

from app.db.models.chat_session import ChatSession
from app.db.models.chat_turn import ChatTurn
from app.exceptions.chat_session import ChatSessionRepositoryError
from app.exceptions.chat_turn import ChatTurnRepositoryError
from app.repositories.chat_session import ChatSessionRepository
from app.repositories.chat_turn import ChatTurnRepository

pytestmark = pytest.mark.integration


@pytest.mark.parametrize(
    ('user_id_length', 'is_supported'),
    [(99, True), (100, True), (101, True), (255, True), (256, False)],
)
async def test_chat_session_user_id_boundaries_on_postgresql(
    db_session,
    user_id_length: int,
    is_supported: bool,
) -> None:
    repository = ChatSessionRepository(db_session)
    chat_session = ChatSession(
        session_id=f'session-{user_id_length}',
        user_id='u' * user_id_length,
    )

    if not is_supported:
        with pytest.raises(ChatSessionRepositoryError):
            await repository.save(chat_session)
        return

    saved_session = await repository.save(chat_session)

    assert saved_session.user_id == 'u' * user_id_length


@pytest.mark.parametrize(
    ('user_id_length', 'is_supported'),
    [(99, True), (100, True), (101, True), (255, True), (256, False)],
)
async def test_chat_turn_user_id_boundaries_on_postgresql(
    db_session,
    user_id_length: int,
    is_supported: bool,
) -> None:
    session_repository = ChatSessionRepository(db_session)
    turn_repository = ChatTurnRepository(db_session)
    chat_session = await session_repository.save(
        ChatSession(session_id=f'session-{user_id_length}')
    )
    chat_turn = ChatTurn(
        request_id=f'request-{user_id_length}',
        chat_session_id=chat_session.id,
        sequence_number=1,
        user_id='u' * user_id_length,
        question='Вопрос',
    )

    if not is_supported:
        with pytest.raises(ChatTurnRepositoryError):
            await turn_repository.save(chat_turn)
        return

    saved_turn = await turn_repository.save(chat_turn)

    assert saved_turn.user_id == 'u' * user_id_length
