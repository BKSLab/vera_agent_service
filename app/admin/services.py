from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from app.db.session import async_session_factory
from app.repositories.chat_turn import ChatTurnRepository
from app.repositories.session_feedback import SessionFeedbackRepository
from app.services.dialogue_search import DialogueSearchService


@asynccontextmanager
async def build_dialogue_search_service() -> AsyncIterator[DialogueSearchService]:
    """Собирает сервис поиска для SQLAdmin BaseView."""
    async with async_session_factory() as db_session:
        yield DialogueSearchService(
            chat_turn_repository=ChatTurnRepository(db_session),
            session_feedback_repository=SessionFeedbackRepository(db_session),
        )
