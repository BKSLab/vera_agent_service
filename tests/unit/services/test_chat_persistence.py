from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from app.db.models.chat_session import ChatSession
from app.db.models.chat_turn import STATUS_GENERATION_FAILED, ChatTurn
from app.exceptions.chat_session import ChatSessionAccessDeniedError
from app.repositories.chat_session import ChatSessionRepository
from app.repositories.chat_turn import ChatTurnRepository
from app.services.chat_persistence import (
    START_CLAIMED,
    START_DUPLICATE_TERMINAL,
    ChatPersistenceService,
)


@pytest.mark.asyncio
async def test_start_turn_creates_session_and_processing_turn():
    session_repository = AsyncMock(spec=ChatSessionRepository)
    turn_repository = AsyncMock(spec=ChatTurnRepository)
    saved_session = ChatSession(
        id=uuid4(),
        session_id='session-1',
        user_id='user-1',
    )
    session_repository.lock_or_create_for_turn.return_value = saved_session
    session_repository.touch_for_turn.return_value = saved_session
    turn_repository.get_by_request_id.return_value = None
    turn_repository.get_next_sequence_number.return_value = 1
    service = ChatPersistenceService(session_repository, turn_repository)

    result = await service.start_turn(
        session_id='session-1',
        request_id='request-1',
        user_id='user-1',
        anonymous_token_hash=None,
        question='Вопрос',
        worker_id='worker-1',
        lease_seconds=900.0,
    )

    assert result.outcome == START_CLAIMED
    saved_turn = turn_repository.save.await_args.args[0]
    assert saved_turn.request_id == 'request-1'
    assert saved_turn.question == 'Вопрос'
    assert saved_turn.status == 'processing'


@pytest.mark.asyncio
async def test_start_turn_returns_completed_turn_without_creating_duplicate():
    session_repository = AsyncMock(spec=ChatSessionRepository)
    turn_repository = AsyncMock(spec=ChatTurnRepository)
    chat_session = ChatSession(
        id=uuid4(),
        session_id='session-1',
        anonymous_token_hash='a' * 64,
    )
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
        anonymous_token_hash='a' * 64,
        question='Вопрос',
        worker_id='worker-1',
        lease_seconds=900.0,
    )

    assert result.outcome == START_DUPLICATE_TERMINAL
    assert result.status == 'completed'
    assert result.answer == 'Ответ'
    turn_repository.save.assert_not_awaited()


@pytest.mark.asyncio
async def test_start_turn_rejects_wrong_anonymous_owner():
    session_repository = AsyncMock(spec=ChatSessionRepository)
    turn_repository = AsyncMock(spec=ChatTurnRepository)
    session_repository.lock_or_create_for_turn.return_value = ChatSession(
        id=uuid4(),
        session_id='session-1',
        anonymous_token_hash='a' * 64,
    )
    turn_repository.get_by_request_id.return_value = None
    service = ChatPersistenceService(session_repository, turn_repository)

    with pytest.raises(ChatSessionAccessDeniedError):
        await service.start_turn(
            session_id='session-1',
            request_id='request-2',
            user_id=None,
            anonymous_token_hash='b' * 64,
            question='Вопрос',
            worker_id='worker-1',
            lease_seconds=900.0,
        )


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
        status=STATUS_GENERATION_FAILED,
        safe_error='Не удалось обработать запрос',
        answer=None,
        latency_ms=250,
        terminal_detail='Сервис временно недоступен, попробуйте позже.',
    )

    turn_repository.fail.assert_awaited_once_with(
        chat_turn=chat_turn,
        status=STATUS_GENERATION_FAILED,
        terminal_detail='Сервис временно недоступен, попробуйте позже.',
        safe_error='Не удалось обработать запрос',
        answer=None,
        latency_ms=250,
    )
