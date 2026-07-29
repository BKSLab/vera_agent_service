from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models.message_feedback import MessageFeedback
from app.exceptions.message_feedback import MessageFeedbackRepositoryError


class MessageFeedbackRepository:
    """Операции PostgreSQL с оценками ответов."""

    def __init__(self, db_session: AsyncSession):
        self.db_session = db_session

    async def get_by_turn_id(self, chat_turn_id) -> MessageFeedback | None:
        """Возвращает оценку конкретной реплики."""
        try:
            result = await self.db_session.execute(
                select(MessageFeedback).where(MessageFeedback.chat_turn_id == chat_turn_id)
            )
            return result.unique().scalar_one_or_none()
        except SQLAlchemyError as error:
            await self.db_session.rollback()
            raise MessageFeedbackRepositoryError(str(error)) from error

    async def save(self, feedback: MessageFeedback) -> MessageFeedback:
        """Создаёт или обновляет оценку ответа."""
        try:
            self.db_session.add(feedback)
            await self.db_session.commit()
            await self.db_session.refresh(feedback)
            return feedback
        except SQLAlchemyError as error:
            await self.db_session.rollback()
            raise MessageFeedbackRepositoryError(str(error)) from error
