from app.db.models.session_feedback import SessionFeedback
from app.exceptions.chat_session import (
    ChatSessionNotFoundError,
    ChatSessionRepositoryError,
)
from app.exceptions.session_feedback import (
    SessionFeedbackAlreadyExistsError,
    SessionFeedbackRepositoryError,
    SessionFeedbackServiceError,
    SessionFeedbackSubmissionMismatchError,
)
from app.repositories.chat_session import ChatSessionRepository
from app.repositories.session_feedback import SessionFeedbackRepository
from app.services.chat_session_access import ensure_chat_session_access


class SessionFeedbackService:
    """Сохраняет развёрнутые отзывы по существующим сессиям."""

    def __init__(
        self,
        chat_session_repository: ChatSessionRepository,
        session_feedback_repository: SessionFeedbackRepository,
    ):
        self.chat_session_repository = chat_session_repository
        self.session_feedback_repository = session_feedback_repository

    async def create_feedback(
        self,
        session_id: str,
        submission_id: str,
        audience: str | None,
        usefulness: int | None,
        trust: int | None,
        comment: str | None,
        contact_email: str | None,
        user_id: str | None,
        anonymous_token_hash: str | None,
    ) -> SessionFeedback:
        """Создаёт отзыв или возвращает ранее созданный по submission_id."""
        try:
            existing = await self.session_feedback_repository.get_by_submission_id(
                submission_id
            )
            if existing is not None:
                if existing.chat_session.session_id != session_id:
                    raise SessionFeedbackSubmissionMismatchError
                ensure_chat_session_access(
                    existing.chat_session,
                    user_id=user_id,
                    anonymous_token_hash=anonymous_token_hash,
                )
                return existing

            chat_session = await self.chat_session_repository.get_by_session_id(session_id)
            if chat_session is None:
                raise ChatSessionNotFoundError(session_id)
            ensure_chat_session_access(
                chat_session,
                user_id=user_id,
                anonymous_token_hash=anonymous_token_hash,
            )

            return await self.session_feedback_repository.save(
                SessionFeedback(
                    chat_session_id=chat_session.id,
                    submission_id=submission_id,
                    audience=audience,
                    usefulness=usefulness,
                    trust=trust,
                    comment=comment,
                    contact_email=contact_email,
                )
            )
        except SessionFeedbackAlreadyExistsError:
            try:
                existing = await self.session_feedback_repository.get_by_submission_id(
                    submission_id
                )
            except SessionFeedbackRepositoryError as error:
                raise SessionFeedbackServiceError from error
            if existing is None:
                raise SessionFeedbackServiceError from None
            if existing.chat_session.session_id != session_id:
                raise SessionFeedbackSubmissionMismatchError from None
            ensure_chat_session_access(
                existing.chat_session,
                user_id=user_id,
                anonymous_token_hash=anonymous_token_hash,
            )
            return existing
        except (ChatSessionRepositoryError, SessionFeedbackRepositoryError) as error:
            raise SessionFeedbackServiceError from error
