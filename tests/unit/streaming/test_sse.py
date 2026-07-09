import asyncio
import json

import httpx
from fastapi import FastAPI

from app.streaming.session_bus import SessionBus
from app.streaming.sse import create_sse_router


def _build_client(session_bus: SessionBus) -> httpx.AsyncClient:
    app = FastAPI()
    app.include_router(create_sse_router(session_bus))
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url='http://test')


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
        async with client.stream('GET', '/sse/s1') as response:
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


async def test_sse_stream_closes_after_error_event():
    session_bus = SessionBus()

    async def publisher():
        await asyncio.sleep(0.05)
        await session_bus.publish('s1', {'type': 'error', 'detail': 'Сервис недоступен'})

    async with _build_client(session_bus) as client:
        publisher_task = asyncio.create_task(publisher())
        received = []
        async with client.stream('GET', '/sse/s1') as response:
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
        async with client.stream('GET', '/sse/s1') as response:
            async for line in response.aiter_lines():
                if line.startswith('data: '):
                    received.append(json.loads(line.removeprefix('data: ')))

    assert received == [{'type': 'token', 'content': 'Буферизовано'}, {'type': 'done'}]


async def test_two_sessions_do_not_cross_deliver_tokens():
    session_bus = SessionBus()

    async def publisher():
        await asyncio.sleep(0.05)
        await session_bus.publish('session-a', {'type': 'token', 'content': 'A'})
        await session_bus.publish('session-a', {'type': 'done'})
        await session_bus.publish('session-b', {'type': 'token', 'content': 'B'})
        await session_bus.publish('session-b', {'type': 'done'})

    async with _build_client(session_bus) as client:
        publisher_task = asyncio.create_task(publisher())

        async def read_session(session_id: str) -> list[dict]:
            received = []
            async with client.stream('GET', f'/sse/{session_id}') as response:
                async for line in response.aiter_lines():
                    if line.startswith('data: '):
                        received.append(json.loads(line.removeprefix('data: ')))
            return received

        results_a, results_b = await asyncio.gather(read_session('session-a'), read_session('session-b'))
        await publisher_task

    assert results_a == [{'type': 'token', 'content': 'A'}, {'type': 'done'}]
    assert results_b == [{'type': 'token', 'content': 'B'}, {'type': 'done'}]
