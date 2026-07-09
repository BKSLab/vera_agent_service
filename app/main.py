import asyncio
from contextlib import AsyncExitStack, asynccontextmanager

from fastapi import FastAPI
from redis.asyncio import Redis

from app.api.v1.endpoints.health import create_health_router
from app.checkpoint.redis_saver import get_redis_checkpointer
from app.clients.http_client import external_api_http_client
from app.clients.llm import get_chat_model
from app.clients.mcp_client import build_kb_search_tool_proxy, get_mcp_client
from app.core.config_logger import logger
from app.core.settings import get_settings
from app.graph.build import build_graph
from app.messaging.consumer import AgentRequestConsumer
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()

    async with AsyncExitStack() as stack:
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
        # на старте приложения (сервиса ещё не существует, раздел 0 плана;
        # раздел 0.1 — MCP сознательно не входит в этот жёсткий startup-чек).
        kb_search_tool = build_kb_search_tool_proxy(mcp_client)

        graph = build_graph(chat_model, kb_search_tool, settings.mcp).compile(checkpointer=checkpointer)

        consumer = AgentRequestConsumer(
            connection_url=settings.rabbitmq.url_connect,
            queue_name=settings.rabbitmq.rabbitmq_queue,
            dlq_name=settings.rabbitmq.rabbitmq_dlq,
            graph=graph,
            token_sink=session_bus.publish,
        )
        logger.info('🚀 Подключение к RabbitMQ...')
        await asyncio.wait_for(consumer.start(), timeout=STARTUP_TIMEOUT_SECONDS)
        stack.push_async_callback(consumer.stop)
        logger.info('✅ RabbitMQ consumer запущен')

        app.include_router(
            create_health_router(
                consumer=consumer,
                redis_health_client=redis_health_client,
                mcp_client=mcp_client,
                mcp_settings=settings.mcp,
            )
        )

        yield


app = FastAPI(lifespan=lifespan)
app.include_router(create_sse_router(session_bus))
