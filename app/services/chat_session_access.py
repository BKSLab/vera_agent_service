from app.db.models.chat_session import ChatSession
from app.exceptions.chat_session import ChatSessionAccessDeniedError


def ensure_chat_session_access(
    chat_session: ChatSession,
    *,
    user_id: str | None,
    anonymous_token_hash: str | None,
) -> None:
    """Проверяет владельца авторизованной или анонимной сессии."""
    if chat_session.user_id is not None:
        if user_id != chat_session.user_id:
            raise ChatSessionAccessDeniedError
        return

    if (
        chat_session.anonymous_token_hash is None
        or anonymous_token_hash != chat_session.anonymous_token_hash
    ):
        raise ChatSessionAccessDeniedError
