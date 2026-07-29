from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models.chat_session import ChatSession
from app.exceptions.chat_session import ChatSessionRepositoryError


class ChatSessionRepository:
    """Операции PostgreSQL с постоянными сессиями диалога."""

    def __init__(self, db_session: AsyncSession):
        self.db_session = db_session

    async def get_by_session_id(self, session_id: str) -> ChatSession | None:
        """Возвращает сессию по внешнему идентификатору."""
        try:
            result = await self.db_session.execute(
                select(ChatSession).where(ChatSession.session_id == session_id)
            )
            return result.unique().scalar_one_or_none()
        except SQLAlchemyError as error:
            await self.db_session.rollback()
            raise ChatSessionRepositoryError(str(error)) from error

    async def save(self, chat_session: ChatSession) -> ChatSession:
        """Сохраняет новую сессию."""
        try:
            self.db_session.add(chat_session)
            await self.db_session.commit()
            await self.db_session.refresh(chat_session)
            return chat_session
        except SQLAlchemyError as error:
            await self.db_session.rollback()
            raise ChatSessionRepositoryError(str(error)) from error

    async def touch(self, chat_session: ChatSession) -> ChatSession:
        """Обновляет дату последней активности сессии."""
        try:
            chat_session.last_activity_at = datetime.now(UTC)
            await self.db_session.commit()
            return chat_session
        except SQLAlchemyError as error:
            await self.db_session.rollback()
            raise ChatSessionRepositoryError(str(error)) from error
