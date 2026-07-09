from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from langgraph.checkpoint.redis.aio import AsyncRedisSaver

from app.core.settings import RedisSettings

SECONDS_PER_MINUTE = 60


@asynccontextmanager
async def get_redis_checkpointer(settings: RedisSettings) -> AsyncIterator[AsyncRedisSaver]:
    """Контекстный менеджер `AsyncRedisSaver` — LangGraph checkpointer
    поверх Redis (Этап 5, AGENT_VERA_ARCHITECTURE.md раздел "Состояние
    диалога (checkpointer)").

    Требует **Redis Stack** (модуль RediSearch), не обычный `redis`-образ —
    `langgraph-checkpoint-redis` использует полнотекстовые индексы для
    чекпоинтов (`FT._LIST` и т.п.), которых нет в ванильном Redis. Найдено
    эмпирически при реализации этого этапа — см. `docker-compose.yml`,
    сервис `redis` (`redis/redis-stack-server`, не `redis:*-alpine`).

    TTL библиотеки — в минутах (`default_ttl`), настройка проекта
    (`redis_session_ttl_seconds`) — в секундах (понятнее в `.env`), отсюда
    конвертация. `refresh_on_read=True` — активная сессия не истекает по
    TTL, пока ей пользуются; таймер сбрасывается на каждое чтение, что и
    даёт семантику "24 часа **неактивности**" (раздел 6 плана, ранее
    открытый вопрос — значение по умолчанию 86400с/24ч).
    """
    ttl_config = {
        'default_ttl': settings.redis_session_ttl_seconds / SECONDS_PER_MINUTE,
        'refresh_on_read': True,
    }
    async with AsyncRedisSaver.from_conn_string(settings.url_connect, ttl=ttl_config) as checkpointer:
        await checkpointer.asetup()
        yield checkpointer
