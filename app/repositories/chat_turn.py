from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy import func, literal, literal_column, select
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.db.models.chat_session import ChatSession
from app.db.models.chat_turn import ChatTurn
from app.db.models.message_feedback import MessageFeedback
from app.db.models.session_feedback import SessionFeedback
from app.exceptions.chat_turn import ChatTurnAlreadyExistsError, ChatTurnRepositoryError


@dataclass(frozen=True)
class RankedChatTurn:
    """Реплика и её релевантность полнотекстовому запросу."""

    chat_turn: ChatTurn
    rank: float


class ChatTurnRepository:
    """Операции PostgreSQL с репликами диалога."""

    def __init__(self, db_session: AsyncSession):
        self.db_session = db_session

    async def get_by_request_id(self, request_id: str) -> ChatTurn | None:
        """Возвращает реплику по внешнему request_id."""
        try:
            result = await self.db_session.execute(
                select(ChatTurn).where(ChatTurn.request_id == request_id)
            )
            return result.unique().scalar_one_or_none()
        except SQLAlchemyError as error:
            await self.db_session.rollback()
            raise ChatTurnRepositoryError(str(error)) from error

    async def get_next_sequence_number(self, chat_session_id: UUID) -> int:
        """Возвращает следующий порядковый номер реплики в сессии."""
        try:
            current_max = (
                await self.db_session.execute(
                    select(func.max(ChatTurn.sequence_number)).where(
                        ChatTurn.chat_session_id == chat_session_id
                    )
                )
            ).scalar_one()
            return (current_max or 0) + 1
        except SQLAlchemyError as error:
            await self.db_session.rollback()
            raise ChatTurnRepositoryError(str(error)) from error

    async def search(
        self,
        search_query: str | None,
        turn_status: str | None,
        rating: str | None,
        audience: str | None,
        created_from: datetime | None,
        created_to: datetime | None,
        limit: int,
    ) -> list[RankedChatTurn]:
        """Ищет реплики через PostgreSQL FTS и административные фильтры."""
        try:
            if search_query:
                ts_query = func.websearch_to_tsquery(
                    literal_column("'russian'::regconfig"),
                    search_query,
                )
                rank = func.ts_rank_cd(ChatTurn.search_vector, ts_query)
            else:
                ts_query = None
                rank = literal(0.0)

            statement = (
                select(ChatTurn, rank.label('rank'))
                .options(selectinload(ChatTurn.feedback))
                .order_by(rank.desc(), ChatTurn.created_at.desc())
                .limit(limit)
            )
            if ts_query is not None:
                statement = statement.where(ChatTurn.search_vector.op('@@')(ts_query))
            if turn_status:
                statement = statement.where(ChatTurn.status == turn_status)
            if rating == 'none':
                statement = statement.where(~ChatTurn.feedback.has())
            elif rating:
                statement = statement.where(
                    ChatTurn.feedback.has(MessageFeedback.value == rating)
                )
            if audience:
                statement = statement.where(
                    ChatTurn.chat_session.has(
                        ChatSession.feedback_entries.any(
                            SessionFeedback.audience == audience
                        )
                    )
                )
            if created_from:
                statement = statement.where(ChatTurn.created_at >= created_from)
            if created_to:
                statement = statement.where(ChatTurn.created_at < created_to)

            rows = (await self.db_session.execute(statement)).unique().all()
            return [
                RankedChatTurn(chat_turn=chat_turn, rank=float(rank_value or 0))
                for chat_turn, rank_value in rows
            ]
        except SQLAlchemyError as error:
            await self.db_session.rollback()
            raise ChatTurnRepositoryError(str(error)) from error

    async def save(self, chat_turn: ChatTurn) -> ChatTurn:
        """Сохраняет новую реплику."""
        try:
            self.db_session.add(chat_turn)
            await self.db_session.commit()
            await self.db_session.refresh(chat_turn)
            return chat_turn
        except IntegrityError as error:
            await self.db_session.rollback()
            raise ChatTurnAlreadyExistsError from error
        except SQLAlchemyError as error:
            await self.db_session.rollback()
            raise ChatTurnRepositoryError(str(error)) from error

    async def complete(
        self,
        chat_turn: ChatTurn,
        answer: str,
        sources: list,
        technical_metadata: dict,
        latency_ms: int,
    ) -> ChatTurn:
        """Помечает реплику успешно завершённой."""
        try:
            chat_turn.answer = answer
            chat_turn.sources = sources
            chat_turn.technical_metadata = technical_metadata
            chat_turn.status = 'completed'
            chat_turn.safe_error = None
            chat_turn.completed_at = datetime.now(UTC)
            chat_turn.latency_ms = latency_ms
            await self.db_session.commit()
            return chat_turn
        except SQLAlchemyError as error:
            await self.db_session.rollback()
            raise ChatTurnRepositoryError(str(error)) from error

    async def fail(
        self,
        chat_turn: ChatTurn,
        status: str,
        safe_error: str,
        answer: str | None,
        latency_ms: int,
    ) -> ChatTurn:
        """Сохраняет ошибку обработки реплики."""
        try:
            chat_turn.status = status
            chat_turn.safe_error = safe_error
            chat_turn.answer = answer
            chat_turn.completed_at = datetime.now(UTC)
            chat_turn.latency_ms = latency_ms
            await self.db_session.commit()
            return chat_turn
        except SQLAlchemyError as error:
            await self.db_session.rollback()
            raise ChatTurnRepositoryError(str(error)) from error
