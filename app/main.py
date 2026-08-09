import asyncio
from contextlib import AsyncExitStack, asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from redis.asyncio import Redis
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from sqlalchemy import text

from app.admin import create_admin
from app.api.v1.endpoints.chat_history import router as chat_history_router
from app.api.v1.endpoints.health import create_health_router
from app.api.v1.endpoints.message_feedback import router as message_feedback_router
from app.api.v1.endpoints.session_feedback import router as session_feedback_router
from app.checkpoint.redis_saver import get_redis_checkpointer
from app.clients.http_client import external_api_http_client
from app.clients.llm import get_chat_model
from app.clients.mcp_client import (
    build_consultation_email_tool_proxy,
    build_kb_search_tool_proxy,
    get_mcp_client,
)
from app.core.config_logger import logger
from app.core.rate_limit import limiter
from app.core.settings import get_settings
from app.db.session import engine
from app.dependencies.services import build_chat_persistence_service
from app.exceptions.chat_turn import ChatPersistenceServiceError
from app.graph.build import build_graph
from app.messaging.consumer import STALE_TURN_MESSAGE, AgentRequestConsumer
from app.observability.tracing import configure_tracing, shutdown_tracing
from app.streaming.session_bus import SessionBus
from app.streaming.sse import create_sse_router

STARTUP_TIMEOUT_SECONDS: float = 10.0
"""Ограничивает время ожидания подключения к RabbitMQ/Redis при старте —
без него `aio-pika.connect_robust` мог бы ждать бесконечно вместо явного
падения приложения (FASTAPI_PATTERNS.md, раздел 5)."""

# Создаётся сразу, не в lifespan — конструктор SessionBus синхронный, без
# I/O, поэтому SSE-роутер можно подключить сразу при определении app,
# не дожидаясь асинхронной инициализации остальных зависимостей ниже.
session_bus = SessionBus()


async def _reconcile_stale_turns(stale_after_seconds: float) -> None:
    """Закрывает реплики, брошенные упавшим процессом.

    Без этого запись остаётся `processing` навсегда, если сообщение уже было
    подтверждено до сбоя и брокер его не вернёт: пользователь при каждой
    перезагрузке истории видел бы вечное «готовит ответ» (VERA-014).
    Недоступность БД здесь не должна ронять старт — consumer поднимется и
    без очистки, а следующий запуск повторит её.
    """
    try:
        async with build_chat_persistence_service() as service:
            closed = await service.reconcile_stale_turns(
                stale_after_seconds=stale_after_seconds,
                detail=STALE_TURN_MESSAGE,
            )
    except ChatPersistenceServiceError:
        logger.exception('⚠️ Не удалось закрыть брошенные реплики при старте')
        return
    if closed:
        logger.warning('⚠️ Закрыто брошенных реплик при старте: %d', closed)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_tracing(settings.observability)

    try:
        async with AsyncExitStack() as stack:
            stack.push_async_callback(external_api_http_client.aclose)

            logger.info('🚀 Проверка подключения к PostgreSQL...')
            async with engine.connect() as connection:
                await connection.execute(text('SELECT 1'))
            logger.info('✅ PostgreSQL готов')

            logger.info('🚀 Подключение к Redis (LangGraph checkpointer)...')
            checkpointer = await asyncio.wait_for(
                stack.enter_async_context(get_redis_checkpointer(settings.redis)),
                timeout=STARTUP_TIMEOUT_SECONDS,
            )
            logger.info('✅ Redis checkpointer готов')

            redis_health_client = Redis.from_url(settings.redis.url_connect)
            stack.push_async_callback(redis_health_client.aclose)

            chat_model = get_chat_model(httpx_client=external_api_http_client, settings=settings.llm)
            mcp_client = get_mcp_client(settings=settings.mcp)
            # Локальный прокси-тул — не требует доступности MCP Tools Server
            # на старте приложения: обе удалённые тулы резолвятся лениво,
            # а MCP сознательно не входит в жёсткий startup-чек.
            kb_search_tool = build_kb_search_tool_proxy(mcp_client)
            consultation_email_tool = build_consultation_email_tool_proxy(
                mcp_client
            )

            graph = build_graph(
                chat_model,
                kb_search_tool,
                consultation_email_tool,
                settings.mcp,
            ).compile(checkpointer=checkpointer)

            await _reconcile_stale_turns(settings.rabbitmq.turn_stale_after_seconds)

            consumer = AgentRequestConsumer(
                connection_url=settings.rabbitmq.url_connect,
                queue_name=settings.rabbitmq.rabbitmq_queue,
                dlq_name=settings.rabbitmq.rabbitmq_dlq,
                graph=graph,
                token_sink=session_bus.publish,
                persistence_service_factory=build_chat_persistence_service,
                lease_seconds=settings.rabbitmq.turn_lease_seconds,
            )
            logger.info('🚀 Подключение к RabbitMQ...')
            await asyncio.wait_for(consumer.start(), timeout=STARTUP_TIMEOUT_SECONDS)
            stack.push_async_callback(consumer.stop)
            logger.info('✅ RabbitMQ consumer запущен')

            app.include_router(
                create_health_router(
                    consumer=consumer,
                    redis_health_client=redis_health_client,
                    db_engine=engine,
                    mcp_client=mcp_client,
                    mcp_settings=settings.mcp,
                )
            )

            yield
    finally:
        await engine.dispose()
        shutdown_tracing()


app = FastAPI(lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)
app.mount('/static', StaticFiles(directory=Path(__file__).parent / 'static'), name='static')
create_admin(app=app, engine=engine)
app.include_router(create_sse_router(session_bus))
app.include_router(chat_history_router, prefix='/api/v1')
app.include_router(message_feedback_router, prefix='/api/v1')
app.include_router(session_feedback_router, prefix='/api/v1')
