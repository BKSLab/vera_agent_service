import logging

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from redis.asyncio import Redis

from app.clients.mcp_client import get_tools_with_retry
from app.core.settings import McpSettings
from app.exceptions.mcp import McpUnavailableError
from app.messaging.consumer import AgentRequestConsumer

logger = logging.getLogger('vera_agent_service')

MCP_HEALTH_CHECK_TIMEOUT_SECONDS: float = 2.0
"""Короткий таймаут для health-проверки MCP — не должен замедлять ответ
`/health`, даже если MCP Tools Server недоступен (раздел 0.1 плана: MCP —
мягкая зависимость, не входит в жёсткий startup-чек)."""


class HealthStatus(BaseModel):
    """`status`/`rabbitmq`/`redis` — жёсткие: недоступность любого из них
    переводит `status` в `degraded` и код ответа в `503`. `mcp` —
    информационное поле, не влияет ни на `status`, ни на код ответа
    (AGENT_SERVICE_PLAN.md, Этап 8.2 — MCP Tools Server ещё не существует,
    раздел 0, недоступность ожидаема)."""

    status: str
    rabbitmq: str
    redis: str
    mcp: str


def create_health_router(
    consumer: AgentRequestConsumer,
    redis_health_client: Redis,
    mcp_client,
    mcp_settings: McpSettings,
) -> APIRouter:
    router = APIRouter()

    @router.get('/health', response_model=HealthStatus)
    async def health() -> JSONResponse:
        rabbitmq_ok = consumer.is_connected

        try:
            await redis_health_client.ping()
            redis_ok = True
        except Exception as error:  # noqa: BLE001 - любая ошибка Redis = недоступен
            logger.warning('⚠️ Redis health-check неуспешен: %s', error)
            redis_ok = False

        try:
            await get_tools_with_retry(mcp_client, retries=1, timeout_seconds=MCP_HEALTH_CHECK_TIMEOUT_SECONDS)
            mcp_ok = True
        except McpUnavailableError:
            mcp_ok = False

        hard_dependencies_ok = rabbitmq_ok and redis_ok
        body = HealthStatus(
            status='ok' if hard_dependencies_ok else 'degraded',
            rabbitmq='ok' if rabbitmq_ok else 'unavailable',
            redis='ok' if redis_ok else 'unavailable',
            mcp='ok' if mcp_ok else 'unavailable',
        )
        return JSONResponse(
            status_code=200 if hard_dependencies_ok else 503,
            content=body.model_dump(),
        )

    return router
