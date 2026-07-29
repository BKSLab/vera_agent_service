from sqlalchemy import select
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models.session_feedback import SessionFeedback
from app.exceptions.session_feedback import (
    SessionFeedbackAlreadyExistsError,
    SessionFeedbackRepositoryError,
)


class SessionFeedbackRepository:
    """Операции PostgreSQL с развёрнутыми отзывами."""

    def __init__(self, db_session: AsyncSession):
        self.db_session = db_session

    async def get_by_submission_id(self, submission_id: str) -> SessionFeedback | None:
        """Возвращает отзыв по ключу идемпотентности."""
        try:
            result = await self.db_session.execute(
                select(SessionFeedback).where(SessionFeedback.submission_id == submission_id)
            )
            return result.unique().scalar_one_or_none()
        except SQLAlchemyError as error:
            await self.db_session.rollback()
            raise SessionFeedbackRepositoryError(str(error)) from error

    async def save(self, feedback: SessionFeedback) -> SessionFeedback:
        """Сохраняет развёрнутый отзыв."""
        try:
            self.db_session.add(feedback)
            await self.db_session.commit()
            await self.db_session.refresh(feedback)
            return feedback
        except IntegrityError as error:
            await self.db_session.rollback()
            raise SessionFeedbackAlreadyExistsError from error
        except SQLAlchemyError as error:
            await self.db_session.rollback()
            raise SessionFeedbackRepositoryError(str(error)) from error
