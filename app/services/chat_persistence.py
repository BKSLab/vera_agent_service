from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from langgraph.checkpoint.redis.aio import AsyncRedisSaver

from app.core.settings import get_settings
from app.db.models.chat_session import ChatSession
from app.db.models.chat_turn import (
    STATUS_PROCESSING,
    TERMINAL_STATUSES,
    ChatTurn,
)
from app.exceptions.chat_session import (
    ChatSessionInactiveError,
    ChatSessionRepositoryError,
)
from app.exceptions.chat_turn import (
    ChatPersistenceServiceError,
    ChatTurnAlreadyExistsError,
    ChatTurnRepositoryError,
    ChatTurnSessionMismatchError,
)
from app.repositories.chat_session import ChatSessionRepository
from app.repositories.chat_turn import ChatTurnRepository
from app.services.chat_session_access import (
    ensure_chat_session_access,
    is_chat_session_active_in_database,
)
from app.services.chat_session_context import has_live_chat_session_context

START_CLAIMED = 'claimed'
"""Реплика создана или перезахвачена после истёкшей аренды — обрабатываем."""

START_DUPLICATE_IN_PROGRESS = 'duplicate_in_progress'
"""Ту же реплику прямо сейчас держит живая аренда — настоящий дубликат."""

START_DUPLICATE_TERMINAL = 'duplicate_terminal'
"""Реплика уже завершена — повтор обязан воспроизвести сохранённый исход."""


@dataclass(frozen=True)
class TurnStartResult:
    """Результат идемпотентной регистрации входящего запроса."""

    outcome: str
    status: str
    answer: str | None = None
    terminal_detail: str | None = None


class ChatPersistenceService:
    """Сохраняет сессии и реплики на границе RabbitMQ-обработки."""

    def __init__(
        self,
        chat_session_repository: ChatSessionRepository,
        chat_turn_repository: ChatTurnRepository,
        checkpointer: AsyncRedisSaver | None = None,
        session_ttl_seconds: int | None = None,
    ):
        self.chat_session_repository = chat_session_repository
        self.chat_turn_repository = chat_turn_repository
        self.checkpointer = checkpointer
        self.session_ttl_seconds = (
            session_ttl_seconds
            if session_ttl_seconds is not None
            else get_settings().redis.redis_session_ttl_seconds
        )
        if self.session_ttl_seconds <= 0:
            raise ValueError('TTL сессии должен быть положительным')

    async def start_turn(
        self,
        session_id: str,
        request_id: str,
        user_id: str | None,
        anonymous_token_hash: str | None,
        question: str,
        worker_id: str,
        lease_seconds: float,
    ) -> TurnStartResult:
        """Создаёт или перезахватывает processing-реплику без дублей.

        Владение обработкой подтверждается арендой `lease_until`, а не самим
        фактом статуса `processing`: после падения процесса запись осталась бы
        `processing` навсегда, и повторная доставка сообщения RabbitMQ
        отбрасывалась бы как дубликат вместе с потерей запроса.
        """
        try:
            existing_turn = await self.chat_turn_repository.get_by_request_id(request_id)
            if existing_turn is not None:
                return await self._resolve_existing_turn(
                    existing_turn,
                    session_id=session_id,
                    request_id=request_id,
                    user_id=user_id,
                    anonymous_token_hash=anonymous_token_hash,
                    worker_id=worker_id,
                    lease_seconds=lease_seconds,
                )

            chat_session = await self.chat_session_repository.lock_or_create_for_turn(
                session_id=session_id,
                user_id=user_id,
                anonymous_token_hash=anonymous_token_hash,
            )
            ensure_chat_session_access(
                chat_session,
                user_id=user_id,
                anonymous_token_hash=anonymous_token_hash,
            )
            await self._ensure_active_session(chat_session)
            if chat_session.user_id is None and user_id is not None:
                chat_session.user_id = user_id
                chat_session.anonymous_token_hash = None
            chat_session = await self.chat_session_repository.touch_for_turn(
                chat_session
            )

            sequence_number = await self.chat_turn_repository.get_next_sequence_number(
                chat_session.id
            )
            await self.chat_turn_repository.save(
                ChatTurn(
                    request_id=request_id,
                    chat_session_id=chat_session.id,
                    sequence_number=sequence_number,
                    user_id=user_id,
                    question=question,
                    status=STATUS_PROCESSING,
                    lease_until=datetime.now(UTC) + timedelta(seconds=lease_seconds),
                    attempt_count=1,
                    worker_id=worker_id,
                )
            )
            return TurnStartResult(outcome=START_CLAIMED, status=STATUS_PROCESSING)
        except ChatTurnAlreadyExistsError:
            # Гонка: реплику успели создать между проверкой и вставкой.
            # Разбираем её тем же путём, что и найденный ранее дубликат.
            existing_turn = await self.chat_turn_repository.get_by_request_id(request_id)
            if existing_turn is None:
                raise ChatPersistenceServiceError from None
            return await self._resolve_existing_turn(
                existing_turn,
                session_id=session_id,
                request_id=request_id,
                user_id=user_id,
                anonymous_token_hash=anonymous_token_hash,
                worker_id=worker_id,
                lease_seconds=lease_seconds,
            )
        except (ChatSessionRepositoryError, ChatTurnRepositoryError) as error:
            raise ChatPersistenceServiceError from error

    async def _ensure_active_session(
        self,
        chat_session: ChatSession,
    ) -> None:
        """Запрещает новую реплику после серверной границы контекста."""
        now = datetime.now(UTC)
        cutoff = now - timedelta(seconds=self.session_ttl_seconds)
        is_active = is_chat_session_active_in_database(
            chat_session,
            cutoff=cutoff,
        )
        if is_active and self.checkpointer is not None:
            try:
                is_active = await has_live_chat_session_context(
                    chat_session,
                    active_at=now,
                    chat_turn_repository=self.chat_turn_repository,
                    checkpointer=self.checkpointer,
                )
            except ChatTurnRepositoryError:
                raise
            except Exception as error:
                raise ChatPersistenceServiceError from error

        if is_active:
            return
        chat_session.closed_at = chat_session.closed_at or now
        await self.chat_session_repository.commit_lifecycle_changes()
        raise ChatSessionInactiveError

    async def _resolve_existing_turn(
        self,
        existing_turn: ChatTurn,
        *,
        session_id: str,
        request_id: str,
        user_id: str | None,
        anonymous_token_hash: str | None,
        worker_id: str,
        lease_seconds: float,
    ) -> TurnStartResult:
        """Решает судьбу повторной доставки уже известного `request_id`."""
        if existing_turn.chat_session.session_id != session_id:
            raise ChatTurnSessionMismatchError
        ensure_chat_session_access(
            existing_turn.chat_session,
            user_id=user_id,
            anonymous_token_hash=anonymous_token_hash,
        )

        if existing_turn.status in TERMINAL_STATUSES:
            return TurnStartResult(
                outcome=START_DUPLICATE_TERMINAL,
                status=existing_turn.status,
                answer=existing_turn.answer,
                terminal_detail=existing_turn.terminal_detail,
            )

        reclaimed = await self.chat_turn_repository.claim_expired_lease(
            request_id=request_id,
            worker_id=worker_id,
            lease_seconds=lease_seconds,
        )
        return TurnStartResult(
            outcome=START_CLAIMED if reclaimed else START_DUPLICATE_IN_PROGRESS,
            status=existing_turn.status,
        )

    async def complete_turn(
        self,
        request_id: str,
        answer: str,
        sources: list,
        technical_metadata: dict,
        latency_ms: int,
    ) -> None:
        """Сохраняет фактически отправленный завершённый ответ."""
        try:
            chat_turn = await self.chat_turn_repository.get_by_request_id(request_id)
            if chat_turn is None:
                raise ChatPersistenceServiceError
            await self.chat_turn_repository.complete(
                chat_turn=chat_turn,
                answer=answer,
                sources=sources,
                technical_metadata=technical_metadata,
                latency_ms=latency_ms,
            )
        except ChatTurnRepositoryError as error:
            raise ChatPersistenceServiceError from error

    async def fail_turn(
        self,
        request_id: str,
        status: str,
        safe_error: str,
        answer: str | None,
        latency_ms: int,
        terminal_detail: str | None = None,
    ) -> None:
        """Сохраняет безопасный результат неуспешной обработки."""
        try:
            chat_turn = await self.chat_turn_repository.get_by_request_id(request_id)
            if chat_turn is None:
                raise ChatPersistenceServiceError
            await self.chat_turn_repository.fail(
                chat_turn=chat_turn,
                status=status,
                safe_error=safe_error,
                answer=answer,
                latency_ms=latency_ms,
                terminal_detail=terminal_detail,
            )
        except ChatTurnRepositoryError as error:
            raise ChatPersistenceServiceError from error

    async def reconcile_stale_turns(self, stale_after_seconds: float, detail: str) -> int:
        """Закрывает брошенные `processing` при старте сервиса."""
        try:
            stale_before = datetime.now(UTC) - timedelta(seconds=stale_after_seconds)
            return await self.chat_turn_repository.reconcile_stale(
                stale_before=stale_before,
                detail=detail,
            )
        except ChatTurnRepositoryError as error:
            raise ChatPersistenceServiceError from error
