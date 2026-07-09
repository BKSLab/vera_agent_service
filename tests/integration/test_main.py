"""Интеграционные тесты сборки приложения (Этап 8) — реальные RabbitMQ и
Redis из `docker-compose.yml`, MCP Tools Server сознательно недоступен
(сервиса ещё не существует, см. AGENT_SERVICE_PLAN.md, раздел 0).
"""

import httpx
import pytest

from app.core.settings import get_settings
from app.main import app, lifespan, session_bus

pytestmark = pytest.mark.integration


async def test_app_starts_and_health_reports_hard_dependencies_ok():
    """MCP недоступен, но это не мешает приложению стартовать и не
    переводит /health в 503 (раздел 0.1 — MCP не входит в жёсткий
    startup-чек, недоступность — информационное поле)."""
    async with lifespan(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url='http://test') as client:
            response = await client.get('/health')

    assert response.status_code == 200
    body = response.json()
    assert body['status'] == 'ok'
    assert body['rabbitmq'] == 'ok'
    assert body['redis'] == 'ok'
    assert body['mcp'] == 'unavailable'


async def test_sse_endpoint_is_mounted_and_accepts_connection():
    """Публикуем `done` заранее (буферизация позднего подключения, Этап 7.3)
    — поток сам корректно завершается сразу после подключения, не оставляя
    незакрытое соединение (открытый навсегда стрим без терминального
    события зависал бы на закрытии `httpx.ASGITransport`-клиента — находка
    при отладке этого теста)."""
    async with lifespan(app):
        await session_bus.publish('smoke-test-session', {'type': 'done'})
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url='http://test') as client:
            async with client.stream('GET', '/sse/smoke-test-session') as response:
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
