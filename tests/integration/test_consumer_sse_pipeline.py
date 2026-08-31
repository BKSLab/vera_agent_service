"""Сквозной интеграционный тест Этапов 6+7: реальный RabbitMQ ->
AgentRequestConsumer -> SessionBus (как TokenSink) -> реальный SSE-клиент
по HTTP. Подтверждает, что `SessionBus.publish` действительно совместим с
`TokenSink`-протоколом, которым Этап 6 был спроектирован независимо от
Этапа 7 (см. AGENT_SERVICE_PLAN.md, "Фактически сделано" Этапа 6).
"""

import asyncio
import json
import time
import uuid
from types import SimpleNamespace

import aio_pika
import httpx
import pytest
from fastapi import FastAPI
from langchain_core.messages import AIMessage

from app.core.settings import get_settings
from app.messaging.consumer import AgentRequestConsumer
from app.streaming.session_bus import SessionBus
from app.streaming.sse import create_sse_router
from app.streaming.ticket import StreamTicketVerifier
from tests.fixtures.stream_ticket import create_stream_ticket

pytestmark = pytest.mark.integration


class _FakeGraph:
    def __init__(self, events: list):
        self._events = events

    def astream_events(self, state, config, version='v2'):
        async def _generator():
            for item in self._events:
                yield item

        return _generator()


def _raw_model_event(content: str) -> dict:
    return {
        'event': 'on_chat_model_stream',
        'metadata': {'langgraph_node': 'generate_direct'},
        'data': {'chunk': SimpleNamespace(content=content)},
    }


def _final_event(content: str) -> dict:
    return {
        'event': 'on_chain_end',
        'parent_ids': ['graph-run'],
        'metadata': {'langgraph_node': 'generate_direct'},
        'data': {'output': {'messages': [AIMessage(content=content)]}},
    }


async def test_message_published_to_rabbitmq_streams_via_real_sse_client():
    settings = get_settings()
    session_id = f'pipeline-{uuid.uuid4().hex[:8]}'
    request_id = f'request-{uuid.uuid4().hex[:8]}'
    suffix = uuid.uuid4().hex[:8]
    queue_name, dlq_name = f'agent.requests.pipeline.{suffix}', f'agent.requests.pipeline.{suffix}.dlq'

    session_bus = SessionBus()
    graph = _FakeGraph(
        [
            _raw_model_event(
                'The user is asking: internal reasoning. Rules check: allowed.'
            ),
            _final_event('Квота составляет 2%.'),
        ]
    )
    consumer = AgentRequestConsumer(
        connection_url=settings.rabbitmq.url_connect,
        queue_name=queue_name,
        dlq_name=dlq_name,
        graph=graph,
        token_sink=session_bus.publish,
    )
    await consumer.start()

    stream_api_key = 'pipeline-test-key'
    ticket = create_stream_ticket(
        api_key=stream_api_key,
        request_id=request_id,
        session_id=session_id,
        user_id='integration-user',
        expires_at=int(time.time()) + 60,
    )
    app = FastAPI()
    app.include_router(
        create_sse_router(
            session_bus,
            StreamTicketVerifier(stream_api_key),
        )
    )
    http_client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test')

    try:
        received: list[dict] = []
        event_ids: list[int] = []

        async def read_sse():
            async with http_client.stream(
                'GET',
                f'/sse/{request_id}?ticket={ticket}',
            ) as response:
                async for line in response.aiter_lines():
                    if line.startswith('id: '):
                        event_ids.append(int(line.removeprefix('id: ')))
                    if line.startswith('data: '):
                        received.append(json.loads(line.removeprefix('data: ')))

        reader_task = asyncio.create_task(read_sse())
        await asyncio.sleep(0.1)  # даём SSE-клиенту подписаться до публикации RabbitMQ-сообщения

        connection = await aio_pika.connect_robust(settings.rabbitmq.url_connect)
        async with connection:
            channel = await connection.channel()
            await channel.default_exchange.publish(
                aio_pika.Message(
                    body=json.dumps(
                        {
                            'session_id': session_id,
                            'request_id': request_id,
                            'user_id': 'integration-user',
                            'message': 'Какая квота?',
                        }
                    ).encode()
                ),
                routing_key=queue_name,
            )

        await asyncio.wait_for(reader_task, timeout=5.0)
    finally:
        await consumer.stop()
        await http_client.aclose()

    assert received == [
        {'type': 'token', 'content': 'Квота составляет 2%.'},
        {'type': 'done', 'used_knowledge_base': False},
    ]
    assert event_ids == [1, 2]
    assert sum(event['type'] in ('done', 'error') for event in received) == 1
