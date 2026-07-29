from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from app.db.models.chat_session import ChatSession
from app.db.models.chat_turn import ChatTurn
from app.db.models.message_feedback import MessageFeedback
from app.repositories.chat_turn import ChatTurnRepository
from app.repositories.message_feedback import MessageFeedbackRepository
from app.services.message_feedback import MessageFeedbackService


@pytest.mark.asyncio
async def test_upsert_feedback_changes_value_without_creating_second_record():
    session = ChatSession(id=uuid4(), session_id='session-1')
    turn = ChatTurn(
        id=uuid4(),
        request_id='request-1',
        chat_session=session,
        chat_session_id=session.id,
        sequence_number=1,
        question='Вопрос',
        answer='Ответ',
        status='completed',
    )
    existing_feedback = MessageFeedback(
        id=uuid4(),
        chat_turn_id=turn.id,
        value='down',
    )
    turn_repository = AsyncMock(spec=ChatTurnRepository)
    feedback_repository = AsyncMock(spec=MessageFeedbackRepository)
    turn_repository.get_by_request_id.return_value = turn
    feedback_repository.get_by_turn_id.return_value = existing_feedback
    feedback_repository.save.side_effect = lambda feedback: feedback
    service = MessageFeedbackService(turn_repository, feedback_repository)

    result = await service.upsert_feedback(
        session_id='session-1',
        request_id='request-1',
        value='up',
    )

    assert result is existing_feedback
    assert result.value == 'up'
    feedback_repository.save.assert_awaited_once_with(existing_feedback)
