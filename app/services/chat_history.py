from app.db.models.chat_turn import ChatTurn
from app.exceptions.chat_session import (
    ChatSessionNotFoundError,
    ChatSessionRepositoryError,
    ChatSessionServiceError,
)
from app.exceptions.chat_turn import ChatTurnRepositoryError
from app.repositories.chat_session import ChatSessionRepository
from app.repositories.chat_turn import ChatTurnRepository


class ChatHistoryService:
    """Возвращает сохранённую пользовательскую историю диалога."""

    def __init__(
        self,
        chat_session_repository: ChatSessionRepository,
        chat_turn_repository: ChatTurnRepository,
    ):
        self.chat_session_repository = chat_session_repository
        self.chat_turn_repository = chat_turn_repository

    async def get_history(self, session_id: str) -> list[ChatTurn]:
        """Возвращает реплики существующей сессии по порядку."""
        try:
            chat_session = await self.chat_session_repository.get_by_session_id(
                session_id
            )
            if chat_session is None:
                raise ChatSessionNotFoundError(session_id)

            return await self.chat_turn_repository.list_by_chat_session_id(
                chat_session.id
            )
        except (ChatSessionRepositoryError, ChatTurnRepositoryError) as error:
            raise ChatSessionServiceError from error
