"""Интеграционные тесты сборки приложения (Этап 8) — реальные RabbitMQ и
Redis; доступность MCP Tools Server не является жёстким startup-условием.
"""

import asyncio
import time

import httpx
import pytest
from fastapi import FastAPI
from sqlalchemy.ext.asyncio import AsyncEngine

from app.api.v1.endpoints import health as health_module
from app.core.settings import get_settings
from app.main import app, lifespan, session_bus
from tests.fixtures.stream_ticket import create_stream_ticket

pytestmark = pytest.mark.integration


class _HangingConnection:
    async def __aenter__(self):
        await asyncio.Event().wait()

    async def __aexit__(self, *_args):
        return None


async def test_lifespan_closes_http_client_and_health_checks_have_deadlines(monkeypatch):
    test_app = FastAPI()

    async def healthy_mcp(*_args, **_kwargs):
        return []

    async def hanging_redis_ping(*_args, **_kwargs):
        await asyncio.Event().wait()

    async with lifespan(test_app):
        external_api_http_client = test_app.state.external_api_http_client
        assert external_api_http_client.is_closed is False
        transport = httpx.ASGITransport(app=test_app)
        async with httpx.AsyncClient(transport=transport, base_url='http://test') as client:
            with monkeypatch.context() as patch:
                patch.setattr(
                    health_module,
                    'HARD_DEPENDENCY_HEALTH_CHECK_TIMEOUT_SECONDS',
                    0.05,
                )
                patch.setattr(health_module, 'get_tools_with_retry', healthy_mcp)
                patch.setattr(health_module.Redis, 'ping', hanging_redis_ping)
                started_at = asyncio.get_running_loop().time()
                redis_response = await client.get('/health')
                redis_elapsed = asyncio.get_running_loop().time() - started_at

            with monkeypatch.context() as patch:
                patch.setattr(
                    health_module,
                    'HARD_DEPENDENCY_HEALTH_CHECK_TIMEOUT_SECONDS',
                    0.05,
                )
                patch.setattr(health_module, 'get_tools_with_retry', healthy_mcp)
                patch.setattr(
                    AsyncEngine,
                    'connect',
                    lambda _engine: _HangingConnection(),
                )
                started_at = asyncio.get_running_loop().time()
                database_response = await client.get('/health')
                database_elapsed = asyncio.get_running_loop().time() - started_at

        assert redis_response.status_code == 503
        assert redis_response.json()['redis'] == 'unavailable'
        assert redis_elapsed < 0.5
        assert database_response.status_code == 503
        assert database_response.json()['database'] == 'unavailable'
        assert database_elapsed < 0.5

    assert external_api_http_client.is_closed is True


async def test_app_starts_and_health_reports_hard_dependencies_ok():
    """Статус MCP информационный и не переводит /health в 503."""
    async with lifespan(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url='http://test') as client:
            response = await client.get('/health')

    assert response.status_code == 200
    body = response.json()
    assert body['status'] == 'ok'
    assert body['rabbitmq'] == 'ok'
    assert body['redis'] == 'ok'
    # MCP — информационная зависимость: тест не должен зависеть от того,
    # запущен ли внешний Tools Server в конкретном окружении.
    assert body['mcp'] in {'ok', 'unavailable'}


async def test_sse_endpoint_is_mounted_and_accepts_connection():
    """Публикуем `done` заранее (буферизация позднего подключения, Этап 7.3)
    — поток сам корректно завершается сразу после подключения, не оставляя
    незакрытое соединение (открытый навсегда стрим без терминального
    события зависал бы на закрытии `httpx.ASGITransport`-клиента — находка
    при отладке этого теста)."""
    async with lifespan(app):
        await session_bus.publish('smoke-test-session', {'type': 'done'})
        ticket = create_stream_ticket(
            api_key=get_settings().app.api_key.get_secret_value(),
            request_id='smoke-test-session',
            expires_at=int(time.time()) + 60,
        )
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url='http://test') as client:
            async with client.stream(
                'GET',
                f'/sse/smoke-test-session?ticket={ticket}',
            ) as response:
                assert response.status_code == 200
                assert response.headers['content-type'].startswith('text/event-stream')
                lines = [line async for line in response.aiter_lines() if line.startswith('data: ')]
                assert lines == ['data: {"type": "done"}']


async def test_app_startup_fails_fast_when_rabbitmq_unreachable(monkeypatch):
    """Недоступность RabbitMQ — жёсткая зависимость (раздел 0.1): старт
    приложения должен явно упасть в пределах STARTUP_TIMEOUT_SECONDS, а не
    зависнуть — `aio_pika.connect_robust` без таймаута ждал бы бесконечно."""
    monkeypatch.setenv('RABBITMQ_PORT', '1')  # заведомо недоступный порт
    get_settings.cache_clear()
    try:
        with pytest.raises((TimeoutError, OSError)):
            async with lifespan(app):
                pass
    finally:
        get_settings.cache_clear()  # восстановить реальные настройки для остальных тестов
