from datetime import datetime

from app.db.models.chat_session import ChatSession
from app.exceptions.chat_session import ChatSessionAccessDeniedError

PREVIOUS_ANONYMOUS_HASH_METADATA_KEY = (
    'lifecycle_previous_anonymous_token_hash'
)


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

    previous_hash = (chat_session.service_metadata or {}).get(
        PREVIOUS_ANONYMOUS_HASH_METADATA_KEY
    )
    if anonymous_token_hash is None or anonymous_token_hash not in {
        chat_session.anonymous_token_hash,
        previous_hash,
    }:
        raise ChatSessionAccessDeniedError


def is_chat_session_active_in_database(
    chat_session: ChatSession,
    *,
    cutoff: datetime,
) -> bool:
    """Проверяет незакрытую сессию по единой границе неактивности."""
    return (
        chat_session.closed_at is None
        and chat_session.last_activity_at is not None
        and chat_session.last_activity_at >= cutoff
    )
