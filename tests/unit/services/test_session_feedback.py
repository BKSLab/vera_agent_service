from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from app.db.models.chat_session import ChatSession
from app.db.models.session_feedback import SessionFeedback
from app.repositories.chat_session import ChatSessionRepository
from app.repositories.session_feedback import SessionFeedbackRepository
from app.services.session_feedback import SessionFeedbackService


@pytest.mark.asyncio
async def test_create_feedback_returns_existing_submission_id():
    session = ChatSession(
        id=uuid4(),
        session_id='session-1',
        user_id='user-1',
    )
    existing = SessionFeedback(
        id=uuid4(),
        chat_session_id=session.id,
        chat_session=session,
        submission_id='submission-1',
    )
    session_repository = AsyncMock(spec=ChatSessionRepository)
    feedback_repository = AsyncMock(spec=SessionFeedbackRepository)
    feedback_repository.get_by_submission_id.return_value = existing
    service = SessionFeedbackService(session_repository, feedback_repository)

    result = await service.create_feedback(
        session_id='session-1',
        submission_id='submission-1',
        audience=None,
        usefulness=None,
        trust=None,
        comment=None,
        contact_email=None,
        user_id='user-1',
        anonymous_token_hash=None,
    )

    assert result is existing
    feedback_repository.save.assert_not_awaited()
