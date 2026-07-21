import asyncio

from app.observability.request_trace import (
    AgentRequestTraceData,
    get_request_trace,
    reset_request_trace,
    set_request_trace,
)


async def test_request_trace_does_not_leak_between_parallel_tasks():
    ready = asyncio.Event()
    observed: dict[str, str] = {}

    async def worker(name: str) -> None:
        token = set_request_trace(AgentRequestTraceData(route=name))
        try:
            if len(observed) == 0:
                ready.set()
            await ready.wait()
            await asyncio.sleep(0)
            observed[name] = get_request_trace().route
        finally:
            reset_request_trace(token)

    await asyncio.gather(worker('direct'), worker('knowledge_base'))

    assert observed == {'direct': 'direct', 'knowledge_base': 'knowledge_base'}
    assert get_request_trace() is None


def test_request_trace_reset_restores_empty_context_between_sequential_requests():
    first = AgentRequestTraceData(route='direct')
    token = set_request_trace(first)
    reset_request_trace(token)

    second = AgentRequestTraceData()
    second_token = set_request_trace(second)
    try:
        assert get_request_trace() is second
        assert second.route == 'unknown'
    finally:
        reset_request_trace(second_token)

    assert get_request_trace() is None
