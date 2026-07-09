"""Сквозной интеграционный тест Этапов 6+7: реальный RabbitMQ ->
AgentRequestConsumer -> SessionBus (как TokenSink) -> реальный SSE-клиент
по HTTP. Подтверждает, что `SessionBus.publish` действительно совместим с
`TokenSink`-протоколом, которым Этап 6 был спроектирован независимо от
Этапа 7 (см. AGENT_SERVICE_PLAN.md, "Фактически сделано" Этапа 6).
"""

import asyncio
import json
import uuid
from types import SimpleNamespace

import aio_pika
import httpx
import pytest
from fastapi import FastAPI

from app.core.settings import get_settings
from app.messaging.consumer import AgentRequestConsumer
from app.streaming.session_bus import SessionBus
from app.streaming.sse import create_sse_router

pytestmark = pytest.mark.integration


class _FakeGraph:
    def __init__(self, events: list):
        self._events = events

    def astream_events(self, state, config, version='v2'):
        async def _generator():
            for item in self._events:
                yield item

        return _generator()


def _token_event(content: str) -> dict:
    return {'event': 'on_chat_model_stream', 'data': {'chunk': SimpleNamespace(content=content)}}


async def test_message_published_to_rabbitmq_streams_via_real_sse_client():
    settings = get_settings()
    session_id = f'pipeline-{uuid.uuid4().hex[:8]}'
    suffix = uuid.uuid4().hex[:8]
    queue_name, dlq_name = f'agent.requests.pipeline.{suffix}', f'agent.requests.pipeline.{suffix}.dlq'

    session_bus = SessionBus()
    graph = _FakeGraph([_token_event('Квота '), _token_event('составляет 2%.')])
    consumer = AgentRequestConsumer(
        connection_url=settings.rabbitmq.url_connect,
        queue_name=queue_name,
        dlq_name=dlq_name,
        graph=graph,
        token_sink=session_bus.publish,
    )
    await consumer.start()

    app = FastAPI()
    app.include_router(create_sse_router(session_bus))
    http_client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test')

    try:
        received: list[dict] = []

        async def read_sse():
            async with http_client.stream('GET', f'/sse/{session_id}') as response:
                async for line in response.aiter_lines():
                    if line.startswith('data: '):
                        received.append(json.loads(line.removeprefix('data: ')))

        reader_task = asyncio.create_task(read_sse())
        await asyncio.sleep(0.1)  # даём SSE-клиенту подписаться до публикации RabbitMQ-сообщения

        connection = await aio_pika.connect_robust(settings.rabbitmq.url_connect)
        async with connection:
            channel = await connection.channel()
            await channel.default_exchange.publish(
                aio_pika.Message(
                    body=json.dumps({'session_id': session_id, 'message': 'Какая квота?'}).encode()
                ),
                routing_key=queue_name,
            )

        await asyncio.wait_for(reader_task, timeout=5.0)
    finally:
        await consumer.stop()
        await http_client.aclose()

    assert received == [
        {'type': 'token', 'content': 'Квота '},
        {'type': 'token', 'content': 'составляет 2%.'},
        {'type': 'done'},
    ]
