from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from app.db.models.chat_session import ChatSession
from app.db.models.chat_turn import ChatTurn
from app.exceptions.chat_session import ChatSessionAccessDeniedError
from app.repositories.chat_session import ChatSessionRepository
from app.repositories.chat_turn import ChatTurnRepository
from app.services.chat_history import ChatHistoryService


@pytest.mark.asyncio
async def test_get_current_session_returns_latest_user_session():
    chat_session = ChatSession(
        id=uuid4(),
        session_id='session-1',
        user_id='user-1',
    )
    session_repository = AsyncMock(spec=ChatSessionRepository)
    turn_repository = AsyncMock(spec=ChatTurnRepository)
    session_repository.get_current_by_user_id.return_value = chat_session
    service = ChatHistoryService(session_repository, turn_repository)

    result = await service.get_current_session('user-1')

    assert result is chat_session
    session_repository.get_current_by_user_id.assert_awaited_once_with(
        'user-1'
    )


@pytest.mark.asyncio
async def test_chat_history_service_returns_session_turns():
    chat_session = ChatSession(
        id=uuid4(),
        session_id='session-1',
        user_id='user-1',
    )
    turns = [
        ChatTurn(
            id=uuid4(),
            request_id='request-1',
            chat_session_id=chat_session.id,
            sequence_number=1,
            question='Вопрос',
            answer='Ответ',
            status='completed',
        )
    ]
    session_repository = AsyncMock(spec=ChatSessionRepository)
    turn_repository = AsyncMock(spec=ChatTurnRepository)
    session_repository.get_by_session_id.return_value = chat_session
    turn_repository.list_by_chat_session_id.return_value = (turns, False)
    service = ChatHistoryService(session_repository, turn_repository)

    result = await service.get_history(
        'session-1',
        user_id='user-1',
        anonymous_token_hash=None,
        limit=30,
        before_sequence=None,
    )

    assert result.turns == turns
    assert result.next_before_sequence is None
    turn_repository.list_by_chat_session_id.assert_awaited_once_with(
        chat_session.id,
        limit=30,
        before_sequence=None,
    )


@pytest.mark.asyncio
async def test_get_history_returns_empty_page_for_unknown_session():
    session_repository = AsyncMock(spec=ChatSessionRepository)
    turn_repository = AsyncMock(spec=ChatTurnRepository)
    session_repository.get_by_session_id.return_value = None
    service = ChatHistoryService(session_repository, turn_repository)

    result = await service.get_history(
        'missing-session',
        user_id=None,
        anonymous_token_hash='a' * 64,
        limit=30,
        before_sequence=None,
    )

    assert result.turns == []
    turn_repository.list_by_chat_session_id.assert_not_awaited()


@pytest.mark.asyncio
async def test_get_history_rejects_another_user():
    session_repository = AsyncMock(spec=ChatSessionRepository)
    turn_repository = AsyncMock(spec=ChatTurnRepository)
    session_repository.get_by_session_id.return_value = ChatSession(
        id=uuid4(),
        session_id='session-1',
        user_id='owner-1',
    )
    service = ChatHistoryService(session_repository, turn_repository)

    with pytest.raises(ChatSessionAccessDeniedError):
        await service.get_history(
            'session-1',
            user_id='owner-2',
            anonymous_token_hash=None,
            limit=30,
            before_sequence=None,
        )
