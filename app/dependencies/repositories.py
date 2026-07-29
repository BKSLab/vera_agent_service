from typing import Annotated

from fastapi import Depends

from app.dependencies.db_session import DbSessionDep
from app.repositories.chat_session import ChatSessionRepository
from app.repositories.chat_turn import ChatTurnRepository
from app.repositories.message_feedback import MessageFeedbackRepository
from app.repositories.session_feedback import SessionFeedbackRepository


def get_chat_session_repository(db_session: DbSessionDep) -> ChatSessionRepository:
    return ChatSessionRepository(db_session)


ChatSessionRepositoryDep = Annotated[
    ChatSessionRepository,
    Depends(get_chat_session_repository),
]


def get_chat_turn_repository(db_session: DbSessionDep) -> ChatTurnRepository:
    return ChatTurnRepository(db_session)


ChatTurnRepositoryDep = Annotated[
    ChatTurnRepository,
    Depends(get_chat_turn_repository),
]


def get_message_feedback_repository(
    db_session: DbSessionDep,
) -> MessageFeedbackRepository:
    return MessageFeedbackRepository(db_session)


MessageFeedbackRepositoryDep = Annotated[
    MessageFeedbackRepository,
    Depends(get_message_feedback_repository),
]


def get_session_feedback_repository(
    db_session: DbSessionDep,
) -> SessionFeedbackRepository:
    return SessionFeedbackRepository(db_session)


SessionFeedbackRepositoryDep = Annotated[
    SessionFeedbackRepository,
    Depends(get_session_feedback_repository),
]
