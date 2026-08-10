from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Literal

from langgraph.checkpoint.redis.aio import AsyncRedisSaver

from app.db.models.chat_session import ChatSession
from app.exceptions.chat_session import (
    ChatSessionAccessDeniedError,
    ChatSessionAlreadyClosedError,
    ChatSessionNotFoundError,
    ChatSessionRepositoryError,
    ChatSessionResolutionConflictError,
    ChatSessionServiceError,
)
from app.exceptions.chat_turn import ChatTurnRepositoryError
from app.repositories.chat_session import ChatSessionRepository
from app.repositories.chat_turn import ChatTurnRepository
from app.services.chat_session_access import (
    PREVIOUS_ANONYMOUS_HASH_METADATA_KEY,
    ensure_chat_session_access,
    is_chat_session_active_in_database,
)
from app.services.chat_session_context import has_live_chat_session_context

BOUNDARY_CREATED = 'created'
BOUNDARY_RETAINED = 'retained'
BOUNDARY_EXPIRED = 'expired'

SUCCESSOR_SESSION_ID_METADATA_KEY = 'lifecycle_successor_session_id'
ANONYMOUS_RECOVERY_OPERATION_ID_METADATA_KEY = (
    'lifecycle_anonymous_recovery_operation_id'
)
SUCCESSOR_RECOVERY_PREDECESSOR_ID_METADATA_KEY = (
    'lifecycle_recovery_predecessor_session_id'
)
SUCCESSOR_RECOVERY_DEADLINE_METADATA_KEY = (
    'lifecycle_recovery_deadline_epoch_seconds'
)
SUCCESSOR_RECOVERY_WINDOW_SECONDS = 300


@dataclass(frozen=True)
class ChatSessionResolution:
    """Результат синхронного определения границы диалога."""

    session_id: str
    previous_session_id: str | None
    boundary: Literal['created', 'retained', 'expired']
    session_ttl_seconds: int


@dataclass(frozen=True)
class ChatSessionCreation:
    """Результат явного создания открытой сессии."""

    session_id: str
    session_ttl_seconds: int


@dataclass(frozen=True)
class ChatSessionClosure:
    """Результат явного закрытия сессии."""

    session_id: str
    closed_at: datetime


class ChatSessionLifecycleService:
    """Согласует PostgreSQL-сессию с живым Redis-контекстом."""

    def __init__(
        self,
        chat_session_repository: ChatSessionRepository,
        chat_turn_repository: ChatTurnRepository,
        checkpointer: AsyncRedisSaver,
        session_ttl_seconds: int,
    ):
        if session_ttl_seconds <= 0:
            raise ValueError('TTL сессии должен быть положительным')
        self.chat_session_repository = chat_session_repository
        self.chat_turn_repository = chat_turn_repository
        self.checkpointer = checkpointer
        self.session_ttl_seconds = session_ttl_seconds

    async def create_session(
        self,
        *,
        session_id: str,
        user_id: str | None,
        anonymous_token_hash: str | None,
    ) -> ChatSessionCreation:
        """Явно создаёт сессию с идемпотентным retry для владельца.

        Args:
            session_id: Клиентский уникальный идентификатор новой сессии.
            user_id: Идентификатор авторизованного владельца.
            anonymous_token_hash: Доказательство anonymous-владельца.

        Returns:
            Идентификатор открытой сессии и единый TTL.

        Raises:
            ChatSessionAccessDeniedError: Владелец не подтверждён.
            ChatSessionAlreadyClosedError: Owned ID уже закрыт.
            ChatSessionServiceError: PostgreSQL недоступен.
        """
        if user_id is None and anonymous_token_hash is None:
            raise ChatSessionAccessDeniedError

        try:
            chat_session, created = (
                await self.chat_session_repository.lock_or_create_for_lifecycle(
                    session_id=session_id,
                    user_id=user_id,
                    anonymous_token_hash=anonymous_token_hash,
                )
            )
            if not created:
                ensure_chat_session_access(
                    chat_session,
                    user_id=user_id,
                    anonymous_token_hash=anonymous_token_hash,
                )
                if chat_session.closed_at is not None:
                    raise ChatSessionAlreadyClosedError

            await self.chat_session_repository.commit_lifecycle_changes()
            return ChatSessionCreation(
                session_id=chat_session.session_id,
                session_ttl_seconds=self.session_ttl_seconds,
            )
        except (
            ChatSessionAccessDeniedError,
            ChatSessionAlreadyClosedError,
        ):
            raise
        except ChatSessionRepositoryError as error:
            raise ChatSessionServiceError from error

    async def close_session(
        self,
        *,
        session_id: str,
        user_id: str | None,
        anonymous_token_hash: str | None,
    ) -> ChatSessionClosure:
        """Явно и идемпотентно закрывает owned-сессию.

        Args:
            session_id: Идентификатор закрываемой сессии.
            user_id: Идентификатор авторизованного владельца.
            anonymous_token_hash: Доказательство anonymous-владельца.

        Returns:
            Идентификатор и сохранённый момент закрытия.

        Raises:
            ChatSessionNotFoundError: Сессия не существует.
            ChatSessionAccessDeniedError: Владелец не подтверждён.
            ChatSessionServiceError: PostgreSQL недоступен.
        """
        try:
            chat_session = (
                await self.chat_session_repository.lock_by_session_id(
                    session_id
                )
            )
            if chat_session is None:
                raise ChatSessionNotFoundError(session_id)
            ensure_chat_session_access(
                chat_session,
                user_id=user_id,
                anonymous_token_hash=anonymous_token_hash,
            )
            chat_session.closed_at = (
                chat_session.closed_at or datetime.now(UTC)
            )
            await self.chat_session_repository.commit_lifecycle_changes()
            return ChatSessionClosure(
                session_id=chat_session.session_id,
                closed_at=chat_session.closed_at,
            )
        except (
            ChatSessionAccessDeniedError,
            ChatSessionNotFoundError,
        ):
            raise
        except ChatSessionRepositoryError as error:
            raise ChatSessionServiceError from error

    async def resolve_session(
        self,
        *,
        session_id: str,
        replacement_session_id: str,
        user_id: str | None,
        anonymous_token_hash: str | None,
        refreshed_anonymous_token_hash: str | None,
        replacement_anonymous_token_hash: str | None,
    ) -> ChatSessionResolution:
        """Создаёт, сохраняет или атомарно заменяет границу диалога.

        Args:
            session_id: Текущий внешний идентификатор диалога.
            replacement_session_id: Заранее созданный идентификатор замены.
            user_id: Идентификатор авторизованного пользователя.
            anonymous_token_hash: Доказательство владения текущей сессией.
            refreshed_anonymous_token_hash: Новый hash retained-сессии.
            replacement_anonymous_token_hash: Hash владельца новой сессии.

        Returns:
            Решение о границе и фактический идентификатор сессии.

        Raises:
            ChatSessionAccessDeniedError: Владелец не подтверждён.
            ChatSessionResolutionConflictError: Повтор указал другой successor.
            ChatSessionServiceError: Хранилища недоступны.
        """
        if user_id is None and anonymous_token_hash is None:
            raise ChatSessionAccessDeniedError

        try:
            current, created = (
                await self.chat_session_repository.lock_or_create_for_lifecycle(
                    session_id=session_id,
                    user_id=user_id,
                    anonymous_token_hash=anonymous_token_hash,
                )
            )
            if created:
                await self.chat_session_repository.commit_lifecycle_changes()
                return self._build_resolution(
                    session_id=current.session_id,
                    boundary=BOUNDARY_CREATED,
                )

            previous_proof_retry = self._ensure_lifecycle_access(
                current,
                replacement_session_id=replacement_session_id,
                user_id=user_id,
                anonymous_token_hash=anonymous_token_hash,
                refreshed_anonymous_token_hash=(
                    refreshed_anonymous_token_hash
                ),
            )
            predecessor_was_already_closed = current.closed_at is not None
            if not previous_proof_retry:
                self._clear_successor_recovery(current)
            now = datetime.now(UTC)
            if await self._is_active(current, now=now):
                self._retain_session(
                    current,
                    now=now,
                    replacement_session_id=replacement_session_id,
                    anonymous_token_hash=anonymous_token_hash,
                    refreshed_anonymous_token_hash=(
                        refreshed_anonymous_token_hash
                    ),
                    previous_proof_retry=previous_proof_retry,
                )
                await self.chat_session_repository.commit_lifecycle_changes()
                return self._build_resolution(
                    session_id=current.session_id,
                    boundary=BOUNDARY_RETAINED,
                )

            self._roll_anonymous_hash(
                current,
                replacement_session_id=replacement_session_id,
                anonymous_token_hash=anonymous_token_hash,
                refreshed_anonymous_token_hash=(
                    refreshed_anonymous_token_hash
                ),
                previous_proof_retry=previous_proof_retry,
                require_refreshed=True,
            )
            successor = await self._replace_expired_session(
                current,
                replacement_session_id=replacement_session_id,
                user_id=user_id,
                replacement_anonymous_token_hash=(
                    replacement_anonymous_token_hash
                ),
                predecessor_was_already_closed=(
                    predecessor_was_already_closed
                ),
                now=now,
            )
            await self.chat_session_repository.commit_lifecycle_changes()
            return self._build_resolution(
                session_id=successor.session_id,
                previous_session_id=current.session_id,
                boundary=BOUNDARY_EXPIRED,
            )
        except (
            ChatSessionAccessDeniedError,
            ChatSessionResolutionConflictError,
        ):
            raise
        except (ChatSessionRepositoryError, ChatTurnRepositoryError) as error:
            raise ChatSessionServiceError from error

    async def get_current_user_session(
        self,
        user_id: str,
    ) -> ChatSession | None:
        """Обнаруживает последнюю открытую auth-сессию для resolve.

        Устаревшая сессия возвращается без изменений как predecessor
        candidate. Только последующий обязательный resolve атомарно закрывает
        её и создаёт replacement; повтор потерянного GET поэтому безопасен.

        Args:
            user_id: Идентификатор авторизованного пользователя.

        Returns:
            Последняя открытая сессия либо None для нового пользователя.

        Raises:
            ChatSessionServiceError: Хранилища недоступны.
        """
        try:
            current = await self.chat_session_repository.get_current_by_user_id(
                user_id
            )
            if current is None:
                return None

            now = datetime.now(UTC)
            if not await self._is_active(current, now=now):
                return current

            current.last_activity_at = now
            await self.chat_session_repository.commit_lifecycle_changes()
            return current
        except (ChatSessionRepositoryError, ChatTurnRepositoryError) as error:
            raise ChatSessionServiceError from error

    async def _is_active(
        self,
        chat_session: ChatSession,
        *,
        now: datetime,
    ) -> bool:
        """Проверяет DB TTL и наличие Redis-контекста для истории."""
        cutoff = now - timedelta(seconds=self.session_ttl_seconds)
        if not is_chat_session_active_in_database(
            chat_session,
            cutoff=cutoff,
        ):
            return False
        try:
            return await has_live_chat_session_context(
                chat_session,
                active_at=now,
                chat_turn_repository=self.chat_turn_repository,
                checkpointer=self.checkpointer,
            )
        except ChatTurnRepositoryError:
            raise
        except Exception as error:
            raise ChatSessionServiceError from error

    def _ensure_lifecycle_access(
        self,
        chat_session: ChatSession,
        *,
        replacement_session_id: str,
        user_id: str | None,
        anonymous_token_hash: str | None,
        refreshed_anonymous_token_hash: str | None,
    ) -> bool:
        """Допускает идемпотентный exact retry одной recovery-операции."""
        if chat_session.user_id is not None:
            ensure_chat_session_access(
                chat_session,
                user_id=user_id,
                anonymous_token_hash=None,
            )
            return False

        if (
            chat_session.anonymous_token_hash is not None
            and anonymous_token_hash == chat_session.anonymous_token_hash
        ):
            return False

        previous_hash = chat_session.service_metadata.get(
            PREVIOUS_ANONYMOUS_HASH_METADATA_KEY
        )
        recovery_operation_id = chat_session.service_metadata.get(
            ANONYMOUS_RECOVERY_OPERATION_ID_METADATA_KEY
        )
        if (
            anonymous_token_hash is not None
            and anonymous_token_hash == previous_hash
            and refreshed_anonymous_token_hash is not None
            and refreshed_anonymous_token_hash
            == chat_session.anonymous_token_hash
            and replacement_session_id == recovery_operation_id
        ):
            return True
        raise ChatSessionAccessDeniedError

    def _retain_session(
        self,
        chat_session: ChatSession,
        *,
        now: datetime,
        replacement_session_id: str,
        anonymous_token_hash: str | None,
        refreshed_anonymous_token_hash: str | None,
        previous_proof_retry: bool,
    ) -> None:
        """Синхронизирует DB activity и при необходимости меняет owner hash."""
        chat_session.last_activity_at = now
        self._roll_anonymous_hash(
            chat_session,
            replacement_session_id=replacement_session_id,
            anonymous_token_hash=anonymous_token_hash,
            refreshed_anonymous_token_hash=refreshed_anonymous_token_hash,
            previous_proof_retry=previous_proof_retry,
            require_refreshed=False,
        )

    def _roll_anonymous_hash(
        self,
        chat_session: ChatSession,
        *,
        replacement_session_id: str,
        anonymous_token_hash: str | None,
        refreshed_anonymous_token_hash: str | None,
        previous_proof_retry: bool,
        require_refreshed: bool,
    ) -> None:
        """Атомарно меняет anonymous hash и сохраняет bounded recovery."""
        if chat_session.user_id is not None:
            return
        if refreshed_anonymous_token_hash is None:
            if require_refreshed:
                raise ChatSessionAccessDeniedError
            return
        if (
            refreshed_anonymous_token_hash
            == chat_session.anonymous_token_hash
        ):
            return

        metadata = dict(chat_session.service_metadata)
        if not previous_proof_retry:
            metadata[PREVIOUS_ANONYMOUS_HASH_METADATA_KEY] = (
                anonymous_token_hash
            )
            metadata[ANONYMOUS_RECOVERY_OPERATION_ID_METADATA_KEY] = (
                replacement_session_id
            )
        chat_session.service_metadata = metadata
        chat_session.anonymous_token_hash = refreshed_anonymous_token_hash

    async def _replace_expired_session(
        self,
        current: ChatSession,
        *,
        replacement_session_id: str,
        user_id: str | None,
        replacement_anonymous_token_hash: str | None,
        predecessor_was_already_closed: bool,
        now: datetime,
    ) -> ChatSession:
        """Закрывает current и подготавливает единственный successor."""
        if user_id is None and replacement_anonymous_token_hash is None:
            raise ChatSessionAccessDeniedError

        metadata = dict(current.service_metadata)
        stored_successor_id = metadata.get(
            SUCCESSOR_SESSION_ID_METADATA_KEY
        )
        if stored_successor_id is not None:
            if stored_successor_id != replacement_session_id:
                raise ChatSessionResolutionConflictError
            successor = (
                await self.chat_session_repository.lock_by_session_id(
                    stored_successor_id
                )
            )
            if successor is None:
                raise ChatSessionServiceError
            self._recover_successor_access(
                successor,
                predecessor_session_id=current.session_id,
                user_id=user_id,
                replacement_anonymous_token_hash=(
                    replacement_anonymous_token_hash
                ),
                now=now,
            )
            if successor.closed_at is not None:
                raise ChatSessionResolutionConflictError
            return successor

        successor, created = (
            await self.chat_session_repository.lock_or_create_for_lifecycle(
                session_id=replacement_session_id,
                user_id=user_id,
                anonymous_token_hash=replacement_anonymous_token_hash,
            )
        )
        if not created:
            if not predecessor_was_already_closed:
                raise ChatSessionResolutionConflictError
            self._ensure_reusable_explicit_successor(
                successor,
                predecessor_session_id=current.session_id,
                user_id=user_id,
                replacement_anonymous_token_hash=(
                    replacement_anonymous_token_hash
                ),
            )
            successor.last_activity_at = now

        current.closed_at = current.closed_at or now
        metadata[SUCCESSOR_SESSION_ID_METADATA_KEY] = successor.session_id
        current.service_metadata = metadata
        successor_metadata = dict(successor.service_metadata)
        successor_metadata[
            SUCCESSOR_RECOVERY_PREDECESSOR_ID_METADATA_KEY
        ] = current.session_id
        successor_metadata[SUCCESSOR_RECOVERY_DEADLINE_METADATA_KEY] = (
            now + timedelta(seconds=SUCCESSOR_RECOVERY_WINDOW_SECONDS)
        ).timestamp()
        successor.service_metadata = successor_metadata
        return successor

    def _ensure_reusable_explicit_successor(
        self,
        successor: ChatSession,
        *,
        predecessor_session_id: str,
        user_id: str | None,
        replacement_anonymous_token_hash: str | None,
    ) -> None:
        """Проверяет open owned successor потерянного explicit-create."""
        ensure_chat_session_access(
            successor,
            user_id=user_id,
            anonymous_token_hash=replacement_anonymous_token_hash,
        )
        if successor.closed_at is not None:
            raise ChatSessionResolutionConflictError
        recovery_predecessor_id = successor.service_metadata.get(
            SUCCESSOR_RECOVERY_PREDECESSOR_ID_METADATA_KEY
        )
        if recovery_predecessor_id not in {
            None,
            predecessor_session_id,
        }:
            raise ChatSessionResolutionConflictError

    def _recover_successor_access(
        self,
        successor: ChatSession,
        *,
        predecessor_session_id: str,
        user_id: str | None,
        replacement_anonymous_token_hash: str | None,
        now: datetime,
    ) -> None:
        """Проверяет exact successor credential потерянного ответа."""
        if successor.user_id is not None:
            ensure_chat_session_access(
                successor,
                user_id=user_id,
                anonymous_token_hash=None,
            )
            return
        recovery_predecessor_id = successor.service_metadata.get(
            SUCCESSOR_RECOVERY_PREDECESSOR_ID_METADATA_KEY
        )
        recovery_deadline = successor.service_metadata.get(
            SUCCESSOR_RECOVERY_DEADLINE_METADATA_KEY
        )
        if (
            recovery_predecessor_id != predecessor_session_id
            or isinstance(recovery_deadline, bool)
            or not isinstance(recovery_deadline, (int, float))
            or now.timestamp() > float(recovery_deadline)
        ):
            raise ChatSessionAccessDeniedError
        if (
            replacement_anonymous_token_hash is None
            or replacement_anonymous_token_hash
            != successor.anonymous_token_hash
        ):
            raise ChatSessionAccessDeniedError

    def _clear_successor_recovery(
        self,
        chat_session: ChatSession,
    ) -> None:
        """Закрывает transport recovery после normal owner resolve."""
        metadata = dict(chat_session.service_metadata)
        predecessor_removed = metadata.pop(
            SUCCESSOR_RECOVERY_PREDECESSOR_ID_METADATA_KEY,
            None,
        )
        deadline_removed = metadata.pop(
            SUCCESSOR_RECOVERY_DEADLINE_METADATA_KEY,
            None,
        )
        if predecessor_removed is not None or deadline_removed is not None:
            chat_session.service_metadata = metadata

    def _build_resolution(
        self,
        *,
        session_id: str,
        boundary: Literal['created', 'retained', 'expired'],
        previous_session_id: str | None = None,
    ) -> ChatSessionResolution:
        """Собирает единый публичный lifecycle-ответ."""
        return ChatSessionResolution(
            session_id=session_id,
            previous_session_id=previous_session_id,
            boundary=boundary,
            session_ttl_seconds=self.session_ttl_seconds,
        )
