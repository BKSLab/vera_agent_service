from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models.chat_turn import ChatTurn
from app.exceptions.chat_turn import ChatTurnAlreadyExistsError, ChatTurnRepositoryError


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
