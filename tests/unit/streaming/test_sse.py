import asyncio
import json
import time

import httpx
from fastapi import FastAPI

from app.streaming.session_bus import SessionBus
from app.streaming.sse import _event_stream, create_sse_router
from app.streaming.ticket import StreamTicketVerifier
from tests.fixtures.stream_ticket import create_stream_ticket

API_KEY = 'shared-test-key'


def _build_client(
    session_bus: SessionBus,
    *,
    heartbeat_interval_seconds: float = 15.0,
) -> httpx.AsyncClient:
    app = FastAPI()
    app.include_router(
        create_sse_router(
            session_bus,
            StreamTicketVerifier(API_KEY),
            heartbeat_interval_seconds=heartbeat_interval_seconds,
        )
    )
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
        event_ids = []
        async with client.stream('GET', _stream_path('s1')) as response:
            assert response.status_code == 200
            assert response.headers['content-type'].startswith('text/event-stream')
            async for line in response.aiter_lines():
                if line.startswith('id: '):
                    event_ids.append(int(line.removeprefix('id: ')))
                if line.startswith('data: '):
                    received.append(json.loads(line.removeprefix('data: ')))
        await publisher_task

    assert received == [
        {'type': 'token', 'content': 'При'},
        {'type': 'token', 'content': 'вет'},
        {'type': 'done'},
    ]
    assert event_ids == [1, 2, 3]

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


async def test_sse_reconnect_uses_last_event_id_for_unread_tail():
    session_bus = SessionBus()
    first_queue = session_bus.subscribe('s1')
    await session_bus.publish('s1', {'type': 'token', 'content': 'A'})
    await session_bus.publish('s1', {'type': 'token', 'content': 'B'})
    await session_bus.publish('s1', {'type': 'done'})
    first_event = first_queue.get_nowait()
    session_bus.unsubscribe('s1', first_queue)

    async with _build_client(session_bus) as client:
        received = []
        event_ids = []
        async with client.stream(
            'GET',
            _stream_path('s1'),
            headers={'Last-Event-ID': str(first_event.event_id)},
        ) as response:
            async for line in response.aiter_lines():
                if line.startswith('id: '):
                    event_ids.append(int(line.removeprefix('id: ')))
                if line.startswith('data: '):
                    received.append(json.loads(line.removeprefix('data: ')))

    assert event_ids == [2, 3]
    assert received == [
        {'type': 'token', 'content': 'B'},
        {'type': 'done'},
    ]


async def test_sse_returns_empty_410_when_replay_window_is_gone():
    session_bus = SessionBus(buffer_max_events=2)
    first_queue = session_bus.subscribe('s1')
    for content in ('A', 'B', 'C'):
        await session_bus.publish(
            's1',
            {'type': 'token', 'content': content},
        )
    await session_bus.publish('s1', {'type': 'done'})
    session_bus.unsubscribe('s1', first_queue)

    async with _build_client(session_bus) as client:
        response = await client.get(
            _stream_path('s1'),
            headers={'Last-Event-ID': '1'},
        )

    assert response.status_code == 410
    assert response.content == b''


async def test_sse_returns_204_when_terminal_event_was_already_acknowledged():
    session_bus = SessionBus()
    await session_bus.publish('s1', {'type': 'done'})

    async with _build_client(session_bus) as client:
        response = await client.get(
            _stream_path('s1'),
            headers={'Last-Event-ID': '1'},
        )

    assert response.status_code == 204
    assert response.content == b''


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
    assert first_queue.get_nowait().payload == {
        'type': 'token',
        'content': 'первый жив',
    }
    session_bus.unsubscribe('request-1', first_queue)


async def test_sse_returns_empty_503_when_safe_request_state_is_exhausted():
    session_bus = SessionBus(
        buffer_max_requests=1,
        state_max_entries=1,
    )
    await session_bus.publish(
        'occupied-request',
        {'type': 'token', 'content': 'занято'},
    )

    async with _build_client(session_bus) as client:
        response = await client.get(_stream_path('request-1'))

    assert response.status_code == 503
    assert response.content == b''


async def test_sse_emits_heartbeat_during_long_processing():
    session_bus = SessionBus(request_deadline_seconds=0.2)

    async def publisher():
        await asyncio.sleep(0.04)
        await session_bus.publish('request-1', {'type': 'done'})

    async with _build_client(
        session_bus,
        heartbeat_interval_seconds=0.01,
    ) as client:
        publisher_task = asyncio.create_task(publisher())
        received = []
        event_ids = []
        async with client.stream('GET', _stream_path('request-1')) as response:
            async for line in response.aiter_lines():
                if line.startswith('id: '):
                    event_ids.append(int(line.removeprefix('id: ')))
                if line.startswith('data: '):
                    received.append(json.loads(line.removeprefix('data: ')))
        await publisher_task

    assert any(event['type'] == 'heartbeat' for event in received)
    assert received[-1] == {'type': 'done'}
    assert event_ids == list(range(1, len(event_ids) + 1))


async def test_sse_deadline_emits_one_terminal_error_and_releases_subscriber():
    session_bus = SessionBus(request_deadline_seconds=0.035)

    async with _build_client(
        session_bus,
        heartbeat_interval_seconds=0.01,
    ) as client:
        received = []
        event_ids = []
        async with client.stream('GET', _stream_path('request-1')) as response:
            async for line in response.aiter_lines():
                if line.startswith('id: '):
                    event_ids.append(int(line.removeprefix('id: ')))
                if line.startswith('data: '):
                    received.append(json.loads(line.removeprefix('data: ')))

    terminal_events = [
        event for event in received if event['type'] in ('done', 'error')
    ]
    assert len(terminal_events) == 1
    assert terminal_events[0]['type'] == 'error'
    assert terminal_events[0]['detail']

    await session_bus.publish(
        'request-1',
        {'type': 'token', 'content': 'поздний ответ'},
    )
    await session_bus.publish('request-1', {'type': 'done'})
    replacement_queue = session_bus.subscribe('request-1')
    replayed_terminal = replacement_queue.get_nowait()
    assert replayed_terminal.event_id == event_ids[-1]
    assert replayed_terminal.payload == terminal_events[0]
    assert replacement_queue.empty()
    session_bus.unsubscribe('request-1', replacement_queue)


async def test_sse_stops_after_first_terminal_event():
    session_bus = SessionBus()
    await session_bus.publish('request-1', {'type': 'done'})
    await session_bus.publish(
        'request-1',
        {'type': 'error', 'detail': 'не должен дойти'},
    )

    async with _build_client(session_bus) as client:
        received = []
        async with client.stream('GET', _stream_path('request-1')) as response:
            async for line in response.aiter_lines():
                if line.startswith('data: '):
                    received.append(json.loads(line.removeprefix('data: ')))

    assert received == [{'type': 'done'}]


async def test_token_wins_timeout_race_before_heartbeat(monkeypatch):
    session_bus = SessionBus()
    queue = session_bus.subscribe('request-1')

    async def timeout_after_token(awaitable, *, timeout):
        del timeout
        awaitable.close()
        await session_bus.publish(
            'request-1',
            {'type': 'token', 'content': 'на границе'},
        )
        raise TimeoutError

    monkeypatch.setattr(
        'app.streaming.sse.asyncio.wait_for',
        timeout_after_token,
    )
    stream = _event_stream(
        session_bus,
        'request-1',
        queue,
        heartbeat_interval_seconds=0.01,
    )

    first_wire_event = await anext(stream)
    await stream.aclose()

    assert first_wire_event.startswith('id: 1\n')
    assert json.loads(first_wire_event.split('data: ', 1)[1]) == {
        'type': 'token',
        'content': 'на границе',
    }
