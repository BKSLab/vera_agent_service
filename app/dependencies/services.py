from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends
from langgraph.checkpoint.redis.aio import AsyncRedisSaver

from app.core.settings import get_settings
from app.db.session import async_session_factory
from app.dependencies.checkpoint import RedisCheckpointerDep
from app.dependencies.repositories import (
    ChatSessionRepositoryDep,
    ChatTurnRepositoryDep,
    MessageFeedbackRepositoryDep,
    SessionFeedbackRepositoryDep,
)
from app.repositories.chat_session import ChatSessionRepository
from app.repositories.chat_turn import ChatTurnRepository
from app.services.chat_history import ChatHistoryService
from app.services.chat_persistence import ChatPersistenceService
from app.services.chat_session_lifecycle import ChatSessionLifecycleService
from app.services.message_feedback import MessageFeedbackService
from app.services.session_feedback import SessionFeedbackService


def get_chat_history_service(
    chat_session_repository: ChatSessionRepositoryDep,
    chat_turn_repository: ChatTurnRepositoryDep,
) -> ChatHistoryService:
    return ChatHistoryService(
        chat_session_repository=chat_session_repository,
        chat_turn_repository=chat_turn_repository,
    )


ChatHistoryServiceDep = Annotated[
    ChatHistoryService,
    Depends(get_chat_history_service),
]


def get_chat_session_lifecycle_service(
    chat_session_repository: ChatSessionRepositoryDep,
    chat_turn_repository: ChatTurnRepositoryDep,
    checkpointer: RedisCheckpointerDep,
) -> ChatSessionLifecycleService:
    """Собирает сервис единого жизненного цикла сессии."""
    return ChatSessionLifecycleService(
        chat_session_repository=chat_session_repository,
        chat_turn_repository=chat_turn_repository,
        checkpointer=checkpointer,
        session_ttl_seconds=(
            get_settings().redis.redis_session_ttl_seconds
        ),
    )


ChatSessionLifecycleServiceDep = Annotated[
    ChatSessionLifecycleService,
    Depends(get_chat_session_lifecycle_service),
]


def get_message_feedback_service(
    chat_turn_repository: ChatTurnRepositoryDep,
    message_feedback_repository: MessageFeedbackRepositoryDep,
) -> MessageFeedbackService:
    return MessageFeedbackService(
        chat_turn_repository=chat_turn_repository,
        message_feedback_repository=message_feedback_repository,
    )


MessageFeedbackServiceDep = Annotated[
    MessageFeedbackService,
    Depends(get_message_feedback_service),
]


def get_session_feedback_service(
    chat_session_repository: ChatSessionRepositoryDep,
    session_feedback_repository: SessionFeedbackRepositoryDep,
) -> SessionFeedbackService:
    return SessionFeedbackService(
        chat_session_repository=chat_session_repository,
        session_feedback_repository=session_feedback_repository,
    )


SessionFeedbackServiceDep = Annotated[
    SessionFeedbackService,
    Depends(get_session_feedback_service),
]


@asynccontextmanager
async def build_chat_persistence_service(
    *,
    checkpointer: AsyncRedisSaver | None = None,
) -> AsyncIterator[ChatPersistenceService]:
    """Собирает persistence service для long-lived RabbitMQ consumer."""
    async with async_session_factory() as db_session:
        yield ChatPersistenceService(
            chat_session_repository=ChatSessionRepository(db_session),
            chat_turn_repository=ChatTurnRepository(db_session),
            checkpointer=checkpointer,
            session_ttl_seconds=(
                get_settings().redis.redis_session_ttl_seconds
            ),
        )
