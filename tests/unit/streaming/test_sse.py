import asyncio
import json
import time

import httpx
from fastapi import FastAPI

from app.streaming.session_bus import SessionBus
from app.streaming.sse import create_sse_router
from app.streaming.ticket import StreamTicketVerifier
from tests.fixtures.stream_ticket import create_stream_ticket

API_KEY = 'shared-test-key'


def _build_client(session_bus: SessionBus) -> httpx.AsyncClient:
    app = FastAPI()
    app.include_router(create_sse_router(session_bus, StreamTicketVerifier(API_KEY)))
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url='http://test')


def _stream_path(request_id: str, *, api_key: str = API_KEY) -> str:
    ticket = create_stream_ticket(
        api_key=api_key,
        request_id=request_id,
        expires_at=int(time.time()) + 60,
    )
    return f'/sse/{request_id}?ticket={ticket}'


async def test_sse_stream_delivers_tokens_in_order_and_closes_after_done():
    session_bus = SessionBus()

    async def publisher():
        # Небольшая задержка — клиент должен успеть открыть соединение
        # и подписаться раньше, чем придут события (штатный путь, не
        # буферизация позднего подключения из Этапа 7.3).
        await asyncio.sleep(0.05)
        await session_bus.publish('s1', {'type': 'token', 'content': 'При'})
        await session_bus.publish('s1', {'type': 'token', 'content': 'вет'})
        await session_bus.publish('s1', {'type': 'done'})

    async with _build_client(session_bus) as client:
        publisher_task = asyncio.create_task(publisher())
        received = []
        async with client.stream('GET', _stream_path('s1')) as response:
            assert response.status_code == 200
            assert response.headers['content-type'].startswith('text/event-stream')
            async for line in response.aiter_lines():
                if line.startswith('data: '):
                    received.append(json.loads(line.removeprefix('data: ')))
        await publisher_task

    assert received == [
        {'type': 'token', 'content': 'При'},
        {'type': 'token', 'content': 'вет'},
        {'type': 'done'},
    ]

    replacement_queue = session_bus.subscribe('s1')
    session_bus.unsubscribe('s1', replacement_queue)


async def test_sse_stream_closes_after_error_event():
    session_bus = SessionBus()

    async def publisher():
        await asyncio.sleep(0.05)
        await session_bus.publish('s1', {'type': 'error', 'detail': 'Сервис недоступен'})

    async with _build_client(session_bus) as client:
        publisher_task = asyncio.create_task(publisher())
        received = []
        async with client.stream('GET', _stream_path('s1')) as response:
            async for line in response.aiter_lines():
                if line.startswith('data: '):
                    received.append(json.loads(line.removeprefix('data: ')))
        await publisher_task

    assert received == [{'type': 'error', 'detail': 'Сервис недоступен'}]


async def test_sse_stream_receives_buffered_events_from_late_subscribe():
    """Consumer уже опубликовал события до того, как клиент открыл SSE —
    session_bus их буферизует (Этап 7.3), и они приходят сразу при
    подключении."""
    session_bus = SessionBus()
    await session_bus.publish('s1', {'type': 'token', 'content': 'Буферизовано'})
    await session_bus.publish('s1', {'type': 'done'})

    async with _build_client(session_bus) as client:
        received = []
        async with client.stream('GET', _stream_path('s1')) as response:
            async for line in response.aiter_lines():
                if line.startswith('data: '):
                    received.append(json.loads(line.removeprefix('data: ')))

    assert received == [{'type': 'token', 'content': 'Буферизовано'}, {'type': 'done'}]


async def test_two_request_ids_do_not_cross_deliver_tokens():
    session_bus = SessionBus()

    async def publisher():
        await asyncio.sleep(0.05)
        await session_bus.publish('request-a', {'type': 'token', 'content': 'A'})
        await session_bus.publish('request-a', {'type': 'done'})
        await session_bus.publish('request-b', {'type': 'token', 'content': 'B'})
        await session_bus.publish('request-b', {'type': 'done'})

    async with _build_client(session_bus) as client:
        publisher_task = asyncio.create_task(publisher())

        async def read_request(request_id: str) -> list[dict]:
            received = []
            async with client.stream('GET', _stream_path(request_id)) as response:
                async for line in response.aiter_lines():
                    if line.startswith('data: '):
                        received.append(json.loads(line.removeprefix('data: ')))
            return received

        results_a, results_b = await asyncio.gather(
            read_request('request-a'),
            read_request('request-b'),
        )
        await publisher_task

    assert results_a == [{'type': 'token', 'content': 'A'}, {'type': 'done'}]
    assert results_b == [{'type': 'token', 'content': 'B'}, {'type': 'done'}]


async def test_sse_rejects_missing_invalid_and_foreign_ticket_without_details():
    session_bus = SessionBus()

    async with _build_client(session_bus) as client:
        missing = await client.get('/sse/request-1')
        invalid = await client.get(_stream_path('request-1', api_key='wrong-key'))
        foreign = await client.get(_stream_path('request-2').replace('/request-2?', '/request-1?'))

    for response in (missing, invalid, foreign):
        assert response.status_code == 401
        assert response.content == b''


async def test_second_subscriber_gets_conflict_without_replacing_first():
    session_bus = SessionBus()
    first_queue = session_bus.subscribe('request-1')

    async with _build_client(session_bus) as client:
        response = await client.get(_stream_path('request-1'))

    assert response.status_code == 409
    await session_bus.publish('request-1', {'type': 'token', 'content': 'первый жив'})
    assert first_queue.get_nowait() == {'type': 'token', 'content': 'первый жив'}
    session_bus.unsubscribe('request-1', first_queue)
