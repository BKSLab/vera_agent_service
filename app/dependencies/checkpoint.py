from typing import Annotated

from fastapi import Depends, Request
from langgraph.checkpoint.redis.aio import AsyncRedisSaver


def get_redis_checkpointer_from_app(request: Request) -> AsyncRedisSaver:
    """Возвращает lifespan-managed Redis checkpointer приложения."""
    return request.app.state.checkpointer


RedisCheckpointerDep = Annotated[
    AsyncRedisSaver,
    Depends(get_redis_checkpointer_from_app),
]
