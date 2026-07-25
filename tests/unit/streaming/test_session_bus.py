from app.streaming.session_bus import SessionBus


async def test_publish_delivers_directly_to_subscribed_queue():
    bus = SessionBus()
    queue = bus.subscribe('s1')

    await bus.publish('s1', {'type': 'token', 'content': 'Привет'})

    event = queue.get_nowait()
    assert event == {'type': 'token', 'content': 'Привет'}


async def test_publish_buffers_when_no_subscriber_and_delivers_on_subscribe():
    bus = SessionBus()

    await bus.publish('s1', {'type': 'token', 'content': 'A'})
    await bus.publish('s1', {'type': 'token', 'content': 'B'})

    queue = bus.subscribe('s1')

    assert queue.get_nowait() == {'type': 'token', 'content': 'A'}
    assert queue.get_nowait() == {'type': 'token', 'content': 'B'}


async def test_buffered_events_expire_after_buffer_window(monkeypatch):
    clock = {'now': 0.0}
    monkeypatch.setattr('app.streaming.session_bus.time.monotonic', lambda: clock['now'])

    bus = SessionBus(buffer_seconds=10.0)
    await bus.publish('s1', {'type': 'token', 'content': 'слишком поздно'})

    clock['now'] = 11.0  # 11с > буфера в 10с
    queue = bus.subscribe('s1')

    assert queue.empty()


async def test_buffered_events_within_window_are_delivered(monkeypatch):
    clock = {'now': 0.0}
    monkeypatch.setattr('app.streaming.session_bus.time.monotonic', lambda: clock['now'])

    bus = SessionBus(buffer_seconds=10.0)
    await bus.publish('s1', {'type': 'token', 'content': 'вовремя'})

    clock['now'] = 5.0  # 5с < буфера в 10с
    queue = bus.subscribe('s1')

    assert queue.get_nowait() == {'type': 'token', 'content': 'вовремя'}


async def test_expired_unsubscribed_request_buffer_is_pruned_on_next_publish(monkeypatch):
    clock = {'now': 0.0}
    monkeypatch.setattr('app.streaming.session_bus.time.monotonic', lambda: clock['now'])

    bus = SessionBus(buffer_seconds=10.0)
    await bus.publish('request-old', {'type': 'done'})

    clock['now'] = 11.0
    await bus.publish('request-new', {'type': 'token', 'content': 'Новый ответ'})

    assert 'request-old' not in bus._buffers
    assert 'request-new' in bus._buffers


async def test_resubscribe_replaces_previous_queue():
    """Одно активное соединение на request_id (раздел 0.1) — вторая
    подписка (например вторая вкладка) замещает первую."""
    bus = SessionBus()
    first_queue = bus.subscribe('s1')
    second_queue = bus.subscribe('s1')

    await bus.publish('s1', {'type': 'token', 'content': 'X'})

    assert second_queue.get_nowait() == {'type': 'token', 'content': 'X'}
    assert first_queue.empty()


async def test_unsubscribe_is_noop_if_queue_was_already_replaced():
    bus = SessionBus()
    first_queue = bus.subscribe('s1')
    second_queue = bus.subscribe('s1')

    bus.unsubscribe('s1', first_queue)

    # Вторая (текущая) подписка всё ещё активна и получает публикации.
    await bus.publish('s1', {'type': 'token', 'content': 'жива'})
    assert second_queue.get_nowait() == {'type': 'token', 'content': 'жива'}


async def test_different_request_ids_do_not_share_queues():
    bus = SessionBus()
    queue_a = bus.subscribe('request-a')
    queue_b = bus.subscribe('request-b')

    await bus.publish('request-a', {'type': 'token', 'content': 'A'})
    await bus.publish('request-b', {'type': 'token', 'content': 'B'})

    assert queue_a.get_nowait() == {'type': 'token', 'content': 'A'}
    assert queue_b.get_nowait() == {'type': 'token', 'content': 'B'}
    assert queue_a.empty()
    assert queue_b.empty()


async def test_buffered_previous_request_does_not_reach_next_request():
    bus = SessionBus()
    await bus.publish('request-old', {'type': 'token', 'content': 'Старый ответ'})
    await bus.publish('request-old', {'type': 'done'})

    new_queue = bus.subscribe('request-new')
    await bus.publish('request-new', {'type': 'token', 'content': 'Новый ответ'})

    assert new_queue.get_nowait() == {'type': 'token', 'content': 'Новый ответ'}
    assert new_queue.empty()
