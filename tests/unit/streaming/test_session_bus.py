import asyncio

import pytest

from app.exceptions.streaming import SessionAlreadySubscribedError
from app.streaming.session_bus import (
    SessionBus,
    SessionBusCapacityExceededError,
    SessionReplayUnavailableError,
)


def _next_payload(queue: asyncio.Queue) -> dict:
    return queue.get_nowait().payload


async def test_publish_delivers_directly_to_subscribed_queue():
    bus = SessionBus()
    queue = bus.subscribe('s1')

    await bus.publish('s1', {'type': 'token', 'content': 'Привет'})

    event = queue.get_nowait()
    assert event.event_id == 1
    assert event.payload == {'type': 'token', 'content': 'Привет'}


async def test_publish_buffers_when_no_subscriber_and_delivers_on_subscribe():
    bus = SessionBus()

    await bus.publish('s1', {'type': 'token', 'content': 'A'})
    await bus.publish('s1', {'type': 'token', 'content': 'B'})

    queue = bus.subscribe('s1')

    first = queue.get_nowait()
    second = queue.get_nowait()
    assert (first.event_id, first.payload) == (1, {'type': 'token', 'content': 'A'})
    assert (second.event_id, second.payload) == (2, {'type': 'token', 'content': 'B'})


async def test_buffered_events_expire_after_buffer_window():
    clock = {'now': 0.0}

    bus = SessionBus(
        buffer_seconds=10.0,
        monotonic_clock=lambda: clock['now'],
    )
    await bus.publish('s1', {'type': 'token', 'content': 'слишком поздно'})

    clock['now'] = 11.0  # 11с > буфера в 10с
    queue = bus.subscribe('s1')

    assert _next_payload(queue)['type'] == 'error'


async def test_buffered_events_within_window_are_delivered():
    clock = {'now': 0.0}

    bus = SessionBus(
        buffer_seconds=10.0,
        monotonic_clock=lambda: clock['now'],
    )
    await bus.publish('s1', {'type': 'token', 'content': 'вовремя'})

    clock['now'] = 5.0  # 5с < буфера в 10с
    queue = bus.subscribe('s1')

    assert _next_payload(queue) == {'type': 'token', 'content': 'вовремя'}


async def test_completed_replay_stays_whole_when_oldest_token_expires_first():
    clock = {'now': 0.0}
    bus = SessionBus(
        buffer_seconds=10.0,
        monotonic_clock=lambda: clock['now'],
    )
    await bus.publish('s1', {'type': 'token', 'content': 'A'})
    clock['now'] = 9.0
    await bus.publish('s1', {'type': 'token', 'content': 'B'})
    await bus.publish('s1', {'type': 'done'})

    clock['now'] = 11.0
    queue = bus.subscribe('s1')

    assert [_next_payload(queue) for _ in range(3)] == [
        {'type': 'token', 'content': 'A'},
        {'type': 'token', 'content': 'B'},
        {'type': 'done'},
    ]
    assert queue.empty()


async def test_expired_unsubscribed_request_buffer_is_pruned_on_next_publish():
    clock = {'now': 0.0}

    bus = SessionBus(
        buffer_seconds=10.0,
        monotonic_clock=lambda: clock['now'],
    )
    await bus.publish('request-old', {'type': 'done'})

    clock['now'] = 11.0
    await bus.publish('request-new', {'type': 'token', 'content': 'Новый ответ'})

    assert 'request-old' not in bus._buffers
    assert 'request-new' in bus._buffers


async def test_resubscribe_rejects_second_subscriber_and_keeps_first():
    bus = SessionBus()
    first_queue = bus.subscribe('s1')

    with pytest.raises(SessionAlreadySubscribedError):
        bus.subscribe('s1')

    await bus.publish('s1', {'type': 'token', 'content': 'X'})

    assert _next_payload(first_queue) == {'type': 'token', 'content': 'X'}


async def test_unsubscribe_is_noop_for_foreign_queue():
    bus = SessionBus()
    first_queue = bus.subscribe('s1')
    foreign_queue = asyncio.Queue()

    bus.unsubscribe('s1', foreign_queue)

    await bus.publish('s1', {'type': 'token', 'content': 'жива'})
    assert _next_payload(first_queue) == {'type': 'token', 'content': 'жива'}


async def test_unsubscribe_releases_slot_for_next_subscriber():
    bus = SessionBus()
    first_queue = bus.subscribe('s1')
    bus.unsubscribe('s1', first_queue)

    second_queue = bus.subscribe('s1')

    assert second_queue is not first_queue


async def test_reconnect_replays_unread_tail_after_last_event_id():
    bus = SessionBus()
    first_queue = bus.subscribe('s1')
    await bus.publish('s1', {'type': 'token', 'content': 'A'})
    await bus.publish('s1', {'type': 'token', 'content': 'B'})
    await bus.publish('s1', {'type': 'done'})

    delivered = first_queue.get_nowait()
    bus.unsubscribe('s1', first_queue)
    replay_queue = bus.subscribe('s1', last_event_id=delivered.event_id)

    assert [_next_payload(replay_queue) for _ in range(2)] == [
        {'type': 'token', 'content': 'B'},
        {'type': 'done'},
    ]
    assert replay_queue.empty()


async def test_completed_replay_gap_is_rejected_instead_of_returning_done():
    bus = SessionBus(buffer_max_events=2)
    first_queue = bus.subscribe('s1')
    await bus.publish('s1', {'type': 'token', 'content': 'A'})
    await bus.publish('s1', {'type': 'token', 'content': 'B'})
    await bus.publish('s1', {'type': 'token', 'content': 'C'})
    await bus.publish('s1', {'type': 'done'})
    bus.unsubscribe('s1', first_queue)

    with pytest.raises(SessionReplayUnavailableError):
        bus.subscribe('s1', last_event_id=1)

    replay_queue = bus.subscribe('s1', last_event_id=2)
    assert [_next_payload(replay_queue) for _ in range(2)] == [
        {'type': 'token', 'content': 'C'},
        {'type': 'done'},
    ]


async def test_live_replay_gap_does_not_terminalize_processing():
    bus = SessionBus(buffer_max_events=2)
    first_queue = bus.subscribe('s1')
    for content in ('A', 'B', 'C'):
        await bus.publish('s1', {'type': 'token', 'content': content})
    bus.unsubscribe('s1', first_queue)

    with pytest.raises(SessionReplayUnavailableError):
        bus.subscribe('s1', last_event_id=0)

    assert bus._states['s1'].terminal_event is None
    await bus.publish('s1', {'type': 'token', 'content': 'D'})
    assert bus._states['s1'].terminal_event is None


async def test_disconnected_replay_ring_keeps_rolling_for_fresh_cursor():
    bus = SessionBus(buffer_max_events=2)
    first_queue = bus.subscribe('s1')
    for content in ('A', 'B', 'C'):
        await bus.publish('s1', {'type': 'token', 'content': content})
    bus.unsubscribe('s1', first_queue)

    await bus.publish('s1', {'type': 'token', 'content': 'D'})
    replay_queue = bus.subscribe('s1', last_event_id=3)

    replayed = replay_queue.get_nowait()
    assert replayed.event_id == 4
    assert replayed.payload == {'type': 'token', 'content': 'D'}
    assert bus._states['s1'].terminal_event is None


async def test_different_request_ids_do_not_share_queues():
    bus = SessionBus()
    queue_a = bus.subscribe('request-a')
    queue_b = bus.subscribe('request-b')

    await bus.publish('request-a', {'type': 'token', 'content': 'A'})
    await bus.publish('request-b', {'type': 'token', 'content': 'B'})

    event_a = queue_a.get_nowait()
    event_b = queue_b.get_nowait()
    assert (event_a.event_id, event_a.payload) == (
        1,
        {'type': 'token', 'content': 'A'},
    )
    assert (event_b.event_id, event_b.payload) == (
        1,
        {'type': 'token', 'content': 'B'},
    )
    assert queue_a.empty()
    assert queue_b.empty()


async def test_buffered_previous_request_does_not_reach_next_request():
    bus = SessionBus()
    await bus.publish('request-old', {'type': 'token', 'content': 'Старый ответ'})
    await bus.publish('request-old', {'type': 'done'})

    new_queue = bus.subscribe('request-new')
    await bus.publish('request-new', {'type': 'token', 'content': 'Новый ответ'})

    assert _next_payload(new_queue) == {'type': 'token', 'content': 'Новый ответ'}
    assert new_queue.empty()


async def test_slow_subscriber_is_disconnected_with_error_and_publish_continues():
    bus = SessionBus(subscriber_queue_max_events=2, buffer_max_events=2)
    slow_queue = bus.subscribe('request-1')
    await bus.publish('request-1', {'type': 'token', 'content': 'A'})
    await bus.publish('request-1', {'type': 'token', 'content': 'B'})

    await bus.publish('request-1', {'type': 'token', 'content': 'C'})

    overflow = slow_queue.get_nowait()
    assert slow_queue.qsize() == 0
    assert overflow.payload['type'] == 'error'
    assert overflow.completes_request is False

    with pytest.raises(SessionAlreadySubscribedError):
        bus.subscribe('request-1')

    await bus.publish('request-1', {'type': 'token', 'content': 'C'})
    await bus.publish('request-1', {'type': 'done'})
    bus.unsubscribe('request-1', slow_queue)
    replacement_queue = bus.subscribe('request-1')

    replayed_terminal = replacement_queue.get_nowait()
    assert replayed_terminal == overflow
    assert replacement_queue.empty()


async def test_late_connect_buffer_overflow_becomes_one_terminal_error():
    bus = SessionBus(buffer_max_events=2)

    await bus.publish('request-1', {'type': 'token', 'content': 'A'})
    await bus.publish('request-1', {'type': 'token', 'content': 'B'})
    await bus.publish('request-1', {'type': 'done'})

    queue = bus.subscribe('request-1')
    assert _next_payload(queue)['type'] == 'error'
    assert queue.empty()


async def test_late_connect_buffers_evict_oldest_request_at_global_limit():
    bus = SessionBus(buffer_max_requests=2)

    await bus.publish('request-1', {'type': 'token', 'content': 'A'})
    await bus.publish('request-2', {'type': 'token', 'content': 'B'})
    await bus.publish('request-3', {'type': 'token', 'content': 'C'})

    assert set(bus._buffers) == {'request-2', 'request-3'}
    evicted_queue = bus.subscribe('request-1')
    assert _next_payload(evicted_queue)['type'] == 'error'


async def test_request_state_has_hard_cap_and_rejects_unsafe_eviction():
    bus = SessionBus(
        buffer_max_requests=1,
        state_max_entries=2,
    )

    await bus.publish('request-1', {'type': 'token', 'content': 'A'})
    await bus.publish('request-2', {'type': 'token', 'content': 'B'})
    await bus.publish('request-3', {'type': 'token', 'content': 'C'})

    assert len(bus._states) == 2
    assert 'request-3' not in bus._states
    with pytest.raises(SessionBusCapacityExceededError):
        bus.subscribe('request-3')


async def test_unfinished_terminal_tombstone_survives_retention_until_producer_done():
    clock = {'now': 0.0}
    bus = SessionBus(
        buffer_seconds=10.0,
        request_deadline_seconds=5.0,
        monotonic_clock=lambda: clock['now'],
    )
    queue = bus.subscribe('request-1')

    clock['now'] = 6.0
    original_terminal = bus.terminalize_deadline('request-1')
    bus.unsubscribe('request-1', queue)
    clock['now'] = 100.0
    await bus.publish('request-2', {'type': 'token', 'content': 'trigger cleanup'})

    assert bus._states['request-1'].terminal_event == original_terminal
    await bus.publish('request-1', {'type': 'token', 'content': 'late'})
    await bus.publish('request-1', {'type': 'done'})
    replay = bus.subscribe('request-1')
    assert replay.get_nowait() == original_terminal
    assert replay.empty()


async def test_expired_buffer_is_cleaned_by_background_task_without_publish():
    clock = {'now': 0.0}
    bus = SessionBus(
        buffer_seconds=10.0,
        cleanup_interval_seconds=0.005,
        monotonic_clock=lambda: clock['now'],
    )
    await bus.start()
    try:
        await bus.publish('request-1', {'type': 'token', 'content': 'A'})
        clock['now'] = 11.0
        for _ in range(20):
            if 'request-1' not in bus._buffers:
                break
            await asyncio.sleep(0.005)
        assert 'request-1' not in bus._buffers
    finally:
        await bus.stop()

    assert bus._cleanup_task is None
    queue = bus.subscribe('request-1')
    assert _next_payload(queue)['type'] == 'error'


async def test_background_cleanup_does_not_remove_active_subscriber():
    clock = {'now': 0.0}
    bus = SessionBus(
        buffer_seconds=10.0,
        cleanup_interval_seconds=0.005,
        request_deadline_seconds=1000.0,
        monotonic_clock=lambda: clock['now'],
    )
    queue = bus.subscribe('request-1')
    await bus.start()
    try:
        clock['now'] = 11.0
        await asyncio.sleep(0.02)
        assert bus._queues['request-1'] is queue
    finally:
        await bus.stop()


async def test_first_terminal_event_wins_and_is_replayed_with_same_id():
    bus = SessionBus()

    await bus.publish('request-1', {'type': 'done'})
    await bus.publish(
        'request-1',
        {'type': 'error', 'detail': 'duplicate'},
    )

    first_queue = bus.subscribe('request-1')
    terminal = first_queue.get_nowait()
    assert terminal.event_id == 1
    assert terminal.payload == {'type': 'done'}
    assert first_queue.empty()

    bus.unsubscribe('request-1', first_queue)
    replay_queue = bus.subscribe('request-1')
    assert replay_queue.get_nowait() == terminal
    assert replay_queue.empty()


async def test_event_id_survives_buffer_ttl_during_quiet_processing():
    clock = {'now': 0.0}
    bus = SessionBus(
        buffer_seconds=10.0,
        request_deadline_seconds=100.0,
        monotonic_clock=lambda: clock['now'],
    )
    queue = bus.subscribe('request-1')
    heartbeat = bus.create_heartbeat('request-1')
    bus.unsubscribe('request-1', queue)

    clock['now'] = 11.0
    await bus.publish('request-1', {'type': 'token', 'content': 'после паузы'})
    continuation_queue = bus.subscribe('request-1')
    continuation = continuation_queue.get_nowait()

    assert heartbeat.event_id == 1
    assert continuation.event_id == 2


def test_reconnect_does_not_restart_overall_deadline():
    clock = {'now': 0.0}
    bus = SessionBus(
        request_deadline_seconds=10.0,
        monotonic_clock=lambda: clock['now'],
    )
    first_queue = bus.subscribe('request-1')
    bus.unsubscribe('request-1', first_queue)

    clock['now'] = 9.0
    second_queue = bus.subscribe('request-1')

    assert bus.remaining_deadline_seconds('request-1') == 1.0
    bus.unsubscribe('request-1', second_queue)
