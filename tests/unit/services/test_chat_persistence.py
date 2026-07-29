from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from app.db.models.chat_session import ChatSession
from app.db.models.chat_turn import ChatTurn
from app.repositories.chat_session import ChatSessionRepository
from app.repositories.chat_turn import ChatTurnRepository
from app.services.chat_persistence import ChatPersistenceService


@pytest.mark.asyncio
async def test_start_turn_creates_session_and_processing_turn():
    session_repository = AsyncMock(spec=ChatSessionRepository)
    turn_repository = AsyncMock(spec=ChatTurnRepository)
    saved_session = ChatSession(
        id=uuid4(),
        session_id='session-1',
        user_id='user-1',
    )
    session_repository.get_by_session_id.return_value = None
    session_repository.save.return_value = saved_session
    turn_repository.get_by_request_id.return_value = None
    turn_repository.get_next_sequence_number.return_value = 1
    service = ChatPersistenceService(session_repository, turn_repository)

    result = await service.start_turn(
        session_id='session-1',
        request_id='request-1',
        user_id='user-1',
        question='Вопрос',
    )

    assert result.created is True
    saved_turn = turn_repository.save.await_args.args[0]
    assert saved_turn.request_id == 'request-1'
    assert saved_turn.question == 'Вопрос'
    assert saved_turn.status == 'processing'


@pytest.mark.asyncio
async def test_start_turn_returns_completed_turn_without_creating_duplicate():
    session_repository = AsyncMock(spec=ChatSessionRepository)
    turn_repository = AsyncMock(spec=ChatTurnRepository)
    chat_session = ChatSession(id=uuid4(), session_id='session-1')
    turn_repository.get_by_request_id.return_value = ChatTurn(
        id=uuid4(),
        request_id='request-1',
        chat_session=chat_session,
        chat_session_id=chat_session.id,
        sequence_number=1,
        question='Вопрос',
        answer='Ответ',
        status='completed',
    )
    service = ChatPersistenceService(session_repository, turn_repository)

    result = await service.start_turn(
        session_id='session-1',
        request_id='request-1',
        user_id=None,
        question='Вопрос',
    )

    assert result.created is False
    assert result.status == 'completed'
    assert result.answer == 'Ответ'
    turn_repository.save.assert_not_awaited()


@pytest.mark.asyncio
async def test_complete_turn_persists_sent_answer_and_metadata():
    session_repository = AsyncMock(spec=ChatSessionRepository)
    turn_repository = AsyncMock(spec=ChatTurnRepository)
    chat_turn = ChatTurn(
        id=uuid4(),
        request_id='request-1',
        chat_session_id=uuid4(),
        sequence_number=1,
        question='Вопрос',
        status='processing',
    )
    turn_repository.get_by_request_id.return_value = chat_turn
    service = ChatPersistenceService(session_repository, turn_repository)

    await service.complete_turn(
        request_id='request-1',
        answer='Фактически отправленный ответ',
        sources=[{'title': 'Источник'}],
        technical_metadata={'route': 'search'},
        latency_ms=125,
    )

    turn_repository.complete.assert_awaited_once_with(
        chat_turn=chat_turn,
        answer='Фактически отправленный ответ',
        sources=[{'title': 'Источник'}],
        technical_metadata={'route': 'search'},
        latency_ms=125,
    )


@pytest.mark.asyncio
async def test_fail_turn_persists_safe_failure():
    session_repository = AsyncMock(spec=ChatSessionRepository)
    turn_repository = AsyncMock(spec=ChatTurnRepository)
    chat_turn = ChatTurn(
        id=uuid4(),
        request_id='request-1',
        chat_session_id=uuid4(),
        sequence_number=1,
        question='Вопрос',
        status='processing',
    )
    turn_repository.get_by_request_id.return_value = chat_turn
    service = ChatPersistenceService(session_repository, turn_repository)

    await service.fail_turn(
        request_id='request-1',
        status='failed',
        safe_error='Не удалось обработать запрос',
        answer=None,
        latency_ms=250,
    )

    turn_repository.fail.assert_awaited_once_with(
        chat_turn=chat_turn,
        status='failed',
        safe_error='Не удалось обработать запрос',
        answer=None,
        latency_ms=250,
    )
