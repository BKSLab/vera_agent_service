from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from uuid import UUID

from sqlalchemy import func, literal, literal_column, select, update
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.db.models.chat_session import ChatSession
from app.db.models.chat_turn import (
    STATUS_COMPLETED,
    STATUS_DELIVERY_UNCONFIRMED,
    STATUS_PROCESSING,
    ChatTurn,
)
from app.db.models.message_feedback import MessageFeedback
from app.db.models.session_feedback import SessionFeedback
from app.exceptions.chat_turn import ChatTurnAlreadyExistsError, ChatTurnRepositoryError

REQUEST_ID_CONSTRAINT = 'uq_vera_chat_turns_request_id'


def _get_constraint_name(error: IntegrityError) -> str | None:
    """Извлекает имя PostgreSQL constraint из psycopg/asyncpg adapters."""
    candidates = (error.orig, getattr(error.orig, '__cause__', None))
    for candidate in candidates:
        if candidate is None:
            continue
        constraint_name = getattr(candidate, 'constraint_name', None)
        if constraint_name:
            return constraint_name
        diagnostic = getattr(candidate, 'diag', None)
        constraint_name = getattr(diagnostic, 'constraint_name', None)
        if constraint_name:
            return constraint_name
    return None


@dataclass(frozen=True)
class RankedChatTurn:
    """Реплика и её релевантность полнотекстовому запросу."""

    chat_turn: ChatTurn
    rank: float


@dataclass(frozen=True)
class ChatSessionTurnState:
    """Aggregate-состояние реплик для проверки живого контекста."""

    has_turns: bool
    has_live_processing_turn: bool


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

    async def list_by_chat_session_id(
        self,
        chat_session_id: UUID,
        *,
        limit: int,
        before_sequence: int | None,
    ) -> tuple[list[ChatTurn], bool]:
        """Возвращает страницу реплик с оценками от старых к новым."""
        try:
            query = (
                select(ChatTurn)
                .where(ChatTurn.chat_session_id == chat_session_id)
                .options(selectinload(ChatTurn.feedback))
                .order_by(ChatTurn.sequence_number.desc())
                .limit(limit + 1)
            )
            if before_sequence is not None:
                query = query.where(ChatTurn.sequence_number < before_sequence)
            result = await self.db_session.execute(query)
            descending_turns = list(result.unique().scalars().all())
            has_more = len(descending_turns) > limit
            page = descending_turns[:limit]
            page.reverse()
            return page, has_more
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

    async def get_session_turn_state(
        self,
        chat_session_id: UUID,
        *,
        active_at: datetime,
    ) -> ChatSessionTurnState:
        """Возвращает aggregate истории и processing с живой арендой."""
        try:
            result = await self.db_session.execute(
                select(
                    func.count(ChatTurn.id),
                    func.count(ChatTurn.id).filter(
                        ChatTurn.status == STATUS_PROCESSING,
                        ChatTurn.lease_until.is_not(None),
                        ChatTurn.lease_until >= active_at,
                    ),
                ).where(ChatTurn.chat_session_id == chat_session_id)
            )
            turn_count, processing_count = result.one()
            return ChatSessionTurnState(
                has_turns=turn_count > 0,
                has_live_processing_turn=processing_count > 0,
            )
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
            constraint_name = _get_constraint_name(error)
            if constraint_name == REQUEST_ID_CONSTRAINT:
                raise ChatTurnAlreadyExistsError from error
            raise ChatTurnRepositoryError(
                f'Нарушено ограничение {constraint_name or "unknown"}.'
            ) from error
        except SQLAlchemyError as error:
            await self.db_session.rollback()
            raise ChatTurnRepositoryError(str(error)) from error

    async def claim_expired_lease(
        self,
        request_id: str,
        worker_id: str,
        lease_seconds: float,
    ) -> bool:
        """Атомарно перезахватывает реплику с истёкшей арендой.

        Одним `UPDATE ... WHERE` — проверка и захват не должны разъезжаться:
        два обработчика одного `request_id` иначе оба решат, что аренда
        свободна. Возвращает True, если аренда досталась этому worker.
        """
        try:
            now = datetime.now(UTC)
            result = await self.db_session.execute(
                update(ChatTurn)
                .where(
                    ChatTurn.request_id == request_id,
                    ChatTurn.status == STATUS_PROCESSING,
                    ChatTurn.lease_until.is_not(None),
                    ChatTurn.lease_until < now,
                )
                .values(
                    lease_until=now + timedelta(seconds=lease_seconds),
                    worker_id=worker_id,
                    attempt_count=ChatTurn.attempt_count + 1,
                    started_at=now,
                )
            )
            await self.db_session.commit()
            return result.rowcount == 1
        except SQLAlchemyError as error:
            await self.db_session.rollback()
            raise ChatTurnRepositoryError(str(error)) from error

    async def reconcile_stale(self, stale_before: datetime, detail: str) -> int:
        """Переводит брошенные `processing` в неопределённый исход.

        Реплика остаётся `processing` навсегда, если процесс упал, а RabbitMQ
        не переприслал сообщение (например оно было подтверждено раньше сбоя).
        `stale_before` задаётся с запасом относительно аренды, чтобы не
        обогнать штатную повторную доставку сразу после рестарта.
        """
        try:
            result = await self.db_session.execute(
                update(ChatTurn)
                .where(
                    ChatTurn.status == STATUS_PROCESSING,
                    ChatTurn.lease_until.is_not(None),
                    ChatTurn.lease_until < stale_before,
                )
                .values(
                    status=STATUS_DELIVERY_UNCONFIRMED,
                    safe_error='StaleLeaseReclaimed',
                    terminal_detail=detail,
                    completed_at=datetime.now(UTC),
                    lease_until=None,
                    worker_id=None,
                )
            )
            await self.db_session.commit()
            return result.rowcount
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
        """Помечает реплику успешно завершённой и снимает аренду."""
        try:
            chat_turn.answer = answer
            chat_turn.sources = sources
            chat_turn.technical_metadata = technical_metadata
            chat_turn.status = STATUS_COMPLETED
            chat_turn.safe_error = None
            chat_turn.terminal_detail = None
            chat_turn.completed_at = datetime.now(UTC)
            chat_turn.latency_ms = latency_ms
            chat_turn.lease_until = None
            chat_turn.worker_id = None
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
        terminal_detail: str | None = None,
    ) -> ChatTurn:
        """Сохраняет терминальный неуспех и снимает аренду.

        `terminal_detail` — ровно тот текст, который ушёл пользователю в
        SSE-событии: повторная доставка обязана воспроизвести его дословно,
        а не собирать заново по статусу.
        """
        try:
            chat_turn.status = status
            chat_turn.safe_error = safe_error
            chat_turn.answer = answer
            chat_turn.terminal_detail = terminal_detail
            chat_turn.completed_at = datetime.now(UTC)
            chat_turn.latency_ms = latency_ms
            chat_turn.lease_until = None
            chat_turn.worker_id = None
            await self.db_session.commit()
            return chat_turn
        except SQLAlchemyError as error:
            await self.db_session.rollback()
            raise ChatTurnRepositoryError(str(error)) from error
