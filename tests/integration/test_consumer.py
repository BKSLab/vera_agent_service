"""Интеграционные тесты `AgentRequestConsumer` (Этап 6) на реальном
RabbitMQ из `docker-compose.yml`.
"""

import asyncio
import json
import uuid
from types import SimpleNamespace

import aio_pika
import pytest

from app.core.settings import get_settings
from app.messaging.consumer import AgentRequestConsumer

pytestmark = pytest.mark.integration


class _FakeGraph:
    def __init__(self, events: list):
        self._events = events
        self.call_count = 0

    def astream_events(self, state, config, version='v2'):
        self.call_count += 1

        async def _generator():
            for item in self._events:
                if isinstance(item, Exception):
                    raise item
                yield item

        return _generator()


def _token_event(content: str) -> dict:
    return {'event': 'on_chat_model_stream', 'data': {'chunk': SimpleNamespace(content=content)}}


class _TokenSinkRecorder:
    def __init__(self):
        self.calls: list[tuple[str, dict]] = []
        self._done = asyncio.Event()

    async def __call__(self, session_id: str, event: dict) -> None:
        self.calls.append((session_id, event))
        if event.get('type') in ('done', 'error'):
            self._done.set()

    async def wait_for_terminal_event(self, timeout: float = 5.0) -> None:
        await asyncio.wait_for(self._done.wait(), timeout=timeout)


@pytest.fixture
async def unique_queue_names():
    """Изолированные имена очередей на тест — не мешают друг другу и не
    конфликтуют с очередями будущего реального сервиса на том же брокере."""
    suffix = uuid.uuid4().hex[:8]
    return f'agent.requests.test.{suffix}', f'agent.requests.test.{suffix}.dlq'


async def test_consumer_processes_published_message_end_to_end(unique_queue_names):
    queue_name, dlq_name = unique_queue_names
    settings = get_settings()
    sink = _TokenSinkRecorder()
    graph = _FakeGraph([_token_event('Квота'), _token_event(' составляет 2%.')])
    consumer = AgentRequestConsumer(
        connection_url=settings.rabbitmq.url_connect,
        queue_name=queue_name,
        dlq_name=dlq_name,
        graph=graph,
        token_sink=sink,
    )
    await consumer.start()
    try:
        connection = await aio_pika.connect_robust(settings.rabbitmq.url_connect)
        async with connection:
            channel = await connection.channel()
            await channel.default_exchange.publish(
                aio_pika.Message(
                    body=json.dumps({'session_id': 'integration-session', 'message': 'Какая квота?'}).encode()
                ),
                routing_key=queue_name,
            )

        await sink.wait_for_terminal_event()
    finally:
        await consumer.stop()

    assert sink.calls == [
        ('integration-session', {'type': 'token', 'content': 'Квота'}),
        ('integration-session', {'type': 'token', 'content': ' составляет 2%.'}),
        ('integration-session', {'type': 'done'}),
    ]


async def test_invalid_message_ends_up_in_dead_letter_queue(unique_queue_names):
    queue_name, dlq_name = unique_queue_names
    settings = get_settings()
    sink = _TokenSinkRecorder()
    graph = _FakeGraph([])
    consumer = AgentRequestConsumer(
        connection_url=settings.rabbitmq.url_connect,
        queue_name=queue_name,
        dlq_name=dlq_name,
        graph=graph,
        token_sink=sink,
    )
    await consumer.start()
    try:
        connection = await aio_pika.connect_robust(settings.rabbitmq.url_connect)
        async with connection:
            channel = await connection.channel()
            await channel.default_exchange.publish(
                aio_pika.Message(body=b'not valid json'),
                routing_key=queue_name,
            )

            dlq = await channel.get_queue(dlq_name)
            incoming = await dlq.get(timeout=5.0, fail=False)

        assert incoming is not None
        assert incoming.body == b'not valid json'
        assert graph.call_count == 0
    finally:
        await consumer.stop()
