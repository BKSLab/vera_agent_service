from dataclasses import dataclass

from app.db.models.chat_session import ChatSession
from app.db.models.chat_turn import ChatTurn
from app.exceptions.chat_session import ChatSessionRepositoryError
from app.exceptions.chat_turn import (
    ChatPersistenceServiceError,
    ChatTurnAlreadyExistsError,
    ChatTurnRepositoryError,
    ChatTurnSessionMismatchError,
)
from app.repositories.chat_session import ChatSessionRepository
from app.repositories.chat_turn import ChatTurnRepository


@dataclass(frozen=True)
class TurnStartResult:
    """Результат идемпотентной регистрации входящего запроса."""

    created: bool
    status: str
    answer: str | None = None


class ChatPersistenceService:
    """Сохраняет сессии и реплики на границе RabbitMQ-обработки."""

    def __init__(
        self,
        chat_session_repository: ChatSessionRepository,
        chat_turn_repository: ChatTurnRepository,
    ):
        self.chat_session_repository = chat_session_repository
        self.chat_turn_repository = chat_turn_repository

    async def start_turn(
        self,
        session_id: str,
        request_id: str,
        user_id: str | None,
        question: str,
    ) -> TurnStartResult:
        """Создаёт сессию и processing-реплику без дублей по request_id."""
        try:
            existing_turn = await self.chat_turn_repository.get_by_request_id(request_id)
            if existing_turn is not None:
                if existing_turn.chat_session.session_id != session_id:
                    raise ChatTurnSessionMismatchError
                return TurnStartResult(
                    created=False,
                    status=existing_turn.status,
                    answer=existing_turn.answer,
                )

            chat_session = await self.chat_session_repository.get_by_session_id(session_id)
            if chat_session is None:
                chat_session = await self.chat_session_repository.save(
                    ChatSession(session_id=session_id, user_id=user_id)
                )
            else:
                chat_session = await self.chat_session_repository.touch(chat_session, user_id)

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
                    status='processing',
                )
            )
            return TurnStartResult(created=True, status='processing')
        except ChatTurnAlreadyExistsError:
            existing_turn = await self.chat_turn_repository.get_by_request_id(request_id)
            if existing_turn is None:
                raise ChatPersistenceServiceError from None
            if existing_turn.chat_session.session_id != session_id:
                raise ChatTurnSessionMismatchError from None
            return TurnStartResult(
                created=False,
                status=existing_turn.status,
                answer=existing_turn.answer,
            )
        except (ChatSessionRepositoryError, ChatTurnRepositoryError) as error:
            raise ChatPersistenceServiceError from error

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
            )
        except ChatTurnRepositoryError as error:
            raise ChatPersistenceServiceError from error
