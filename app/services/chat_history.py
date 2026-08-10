from dataclasses import dataclass

from app.db.models.chat_turn import ChatTurn
from app.exceptions.chat_session import (
    ChatSessionAccessDeniedError,
    ChatSessionNotFoundError,
    ChatSessionRepositoryError,
    ChatSessionServiceError,
)
from app.exceptions.chat_turn import ChatTurnRepositoryError
from app.repositories.chat_session import ChatSessionRepository
from app.repositories.chat_turn import ChatTurnRepository
from app.services.chat_session_access import ensure_chat_session_access


@dataclass(frozen=True)
class ChatHistoryPage:
    turns: list[ChatTurn]
    next_before_sequence: int | None


class ChatHistoryService:
    """Возвращает сохранённую пользовательскую историю диалога."""

    def __init__(
        self,
        chat_session_repository: ChatSessionRepository,
        chat_turn_repository: ChatTurnRepository,
    ):
        self.chat_session_repository = chat_session_repository
        self.chat_turn_repository = chat_turn_repository

    async def get_history(
        self,
        session_id: str,
        *,
        user_id: str | None,
        anonymous_token_hash: str | None,
        limit: int,
        before_sequence: int | None,
    ) -> ChatHistoryPage:
        """Возвращает реплики существующей сессии по порядку."""
        try:
            chat_session = await self.chat_session_repository.get_by_session_id(
                session_id
            )
            if chat_session is None:
                raise ChatSessionNotFoundError(session_id)

            ensure_chat_session_access(
                chat_session,
                user_id=user_id,
                anonymous_token_hash=anonymous_token_hash,
            )
            turns, has_more = await self.chat_turn_repository.list_by_chat_session_id(
                chat_session.id,
                limit=limit,
                before_sequence=before_sequence,
            )
            return ChatHistoryPage(
                turns=turns,
                next_before_sequence=(
                    turns[0].sequence_number if has_more and turns else None
                ),
            )
        except (ChatSessionAccessDeniedError, ChatSessionNotFoundError):
            raise
        except (ChatSessionRepositoryError, ChatTurnRepositoryError) as error:
            raise ChatSessionServiceError from error
