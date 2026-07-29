from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import func, literal, literal_column, select
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models.chat_session import ChatSession
from app.db.models.chat_turn import ChatTurn
from app.db.models.message_feedback import MessageFeedback
from app.db.models.session_feedback import SessionFeedback
from app.exceptions.session_feedback import (
    SessionFeedbackAlreadyExistsError,
    SessionFeedbackRepositoryError,
)


@dataclass(frozen=True)
class RankedSessionFeedback:
    """Развёрнутый отзыв и релевантность его комментария запросу."""

    feedback: SessionFeedback
    rank: float


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

    async def search_comments(
        self,
        search_query: str | None,
        turn_status: str | None,
        rating: str | None,
        audience: str | None,
        created_from: datetime | None,
        created_to: datetime | None,
        limit: int,
    ) -> list[RankedSessionFeedback]:
        """Ищет комментарии развёрнутых отзывов через PostgreSQL FTS."""
        try:
            if search_query:
                ts_query = func.websearch_to_tsquery(
                    literal_column("'russian'::regconfig"),
                    search_query,
                )
                rank = func.ts_rank_cd(SessionFeedback.search_vector, ts_query)
            else:
                ts_query = None
                rank = literal(0.0)

            statement = (
                select(SessionFeedback, rank.label('rank'))
                .where(SessionFeedback.comment.is_not(None))
                .order_by(rank.desc(), SessionFeedback.created_at.desc())
                .limit(limit)
            )
            if ts_query is not None:
                statement = statement.where(
                    SessionFeedback.search_vector.op('@@')(ts_query)
                )
            if audience:
                statement = statement.where(SessionFeedback.audience == audience)
            if turn_status:
                statement = statement.where(
                    SessionFeedback.chat_session.has(
                        ChatSession.turns.any(ChatTurn.status == turn_status)
                    )
                )
            if rating == 'none':
                statement = statement.where(
                    SessionFeedback.chat_session.has(
                        ChatSession.turns.any(~ChatTurn.feedback.has())
                    )
                )
            elif rating:
                statement = statement.where(
                    SessionFeedback.chat_session.has(
                        ChatSession.turns.any(
                            ChatTurn.feedback.has(MessageFeedback.value == rating)
                        )
                    )
                )
            if created_from:
                statement = statement.where(SessionFeedback.created_at >= created_from)
            if created_to:
                statement = statement.where(SessionFeedback.created_at < created_to)

            rows = (await self.db_session.execute(statement)).unique().all()
            return [
                RankedSessionFeedback(feedback=feedback, rank=float(rank_value or 0))
                for feedback, rank_value in rows
            ]
        except SQLAlchemyError as error:
            await self.db_session.rollback()
            raise SessionFeedbackRepositoryError(str(error)) from error
