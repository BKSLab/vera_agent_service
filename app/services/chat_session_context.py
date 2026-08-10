from datetime import datetime

from langgraph.checkpoint.redis.aio import AsyncRedisSaver

from app.db.models.chat_session import ChatSession
from app.repositories.chat_turn import ChatTurnRepository


async def has_live_chat_session_context(
    chat_session: ChatSession,
    *,
    active_at: datetime,
    chat_turn_repository: ChatTurnRepository,
    checkpointer: AsyncRedisSaver,
) -> bool:
    """Проверяет Redis-контекст с учётом окна до первого checkpoint.

    Durable processing-реплика появляется в PostgreSQL до первого сохранения
    графа в Redis. В этом коротком окне отсутствие checkpoint ещё не означает
    потерю контекста. Для terminal-истории checkpoint остаётся обязательным.
    """
    turn_state = await chat_turn_repository.get_session_turn_state(
        chat_session.id,
        active_at=active_at,
    )
    if not turn_state.has_turns:
        return True

    checkpoint = await checkpointer.aget_tuple(
        {'configurable': {'thread_id': chat_session.session_id}}
    )
    return checkpoint is not None or turn_state.has_live_processing_turn
