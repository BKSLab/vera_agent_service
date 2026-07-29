from app.db.models.message_feedback import MessageFeedback
from app.exceptions.chat_turn import (
    ChatTurnNotCompletedError,
    ChatTurnNotFoundError,
    ChatTurnRepositoryError,
    ChatTurnSessionMismatchError,
)
from app.exceptions.message_feedback import (
    MessageFeedbackRepositoryError,
    MessageFeedbackServiceError,
)
from app.repositories.chat_turn import ChatTurnRepository
from app.repositories.message_feedback import MessageFeedbackRepository
from app.services.chat_session_access import ensure_chat_session_access


class MessageFeedbackService:
    """Создаёт или изменяет оценку конкретного ответа Веры."""

    def __init__(
        self,
        chat_turn_repository: ChatTurnRepository,
        message_feedback_repository: MessageFeedbackRepository,
    ):
        self.chat_turn_repository = chat_turn_repository
        self.message_feedback_repository = message_feedback_repository

    async def upsert_feedback(
        self,
        session_id: str,
        request_id: str,
        value: str,
        user_id: str | None,
        anonymous_token_hash: str | None,
    ) -> MessageFeedback:
        """Сохраняет единственную оценку завершённой реплики."""
        try:
            chat_turn = await self.chat_turn_repository.get_by_request_id(request_id)
            if chat_turn is None:
                raise ChatTurnNotFoundError(request_id)
            if chat_turn.chat_session.session_id != session_id:
                raise ChatTurnSessionMismatchError
            ensure_chat_session_access(
                chat_turn.chat_session,
                user_id=user_id,
                anonymous_token_hash=anonymous_token_hash,
            )
            if chat_turn.status != 'completed':
                raise ChatTurnNotCompletedError

            feedback = await self.message_feedback_repository.get_by_turn_id(chat_turn.id)
            if feedback is None:
                feedback = MessageFeedback(chat_turn_id=chat_turn.id, value=value)
            else:
                feedback.value = value
            return await self.message_feedback_repository.save(feedback)
        except (ChatTurnRepositoryError, MessageFeedbackRepositoryError) as error:
            raise MessageFeedbackServiceError from error
