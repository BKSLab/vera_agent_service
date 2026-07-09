import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from app.clients.mcp_client import call_tool_with_retry
from app.messaging.consumer import AgentRequestConsumer
from app.observability.tracing import reset_for_tests
from app.streaming.session_bus import SessionBus

# `set_tracer_provider()` можно успешно вызвать только один раз за процесс
# (см. docstring reset_for_tests) — настраиваем провайдер один раз на
# модуль, между тестами только чистим накопленные spans.
_exporter = InMemorySpanExporter()
reset_for_tests(_exporter)


@pytest.fixture(autouse=True)
def _clear_spans():
    _exporter.clear()
    yield


class _FakeTool:
    name = 'kb_search'

    async def ainvoke(self, arguments: dict):
        return {'chunks': []}


class _FakeGraph:
    def astream_events(self, state, config, version='v2'):
        async def _generator():
            return
            yield  # pragma: no cover - делает функцию генератором с пустым потоком

        return _generator()


async def _noop_sink(session_id: str, event: dict) -> None:
    return None


def _span_names() -> list[str]:
    return [span.name for span in _exporter.get_finished_spans()]


async def test_mcp_tool_call_creates_named_span():
    await call_tool_with_retry(_FakeTool(), {'query': 'q'}, retries=1, timeout_seconds=1.0)

    assert 'mcp.tool_call' in _span_names()


async def test_sse_deliver_creates_named_span():
    bus = SessionBus()

    await bus.publish('s1', {'type': 'token', 'content': 'x'})

    assert 'sse.deliver' in _span_names()


async def test_rabbitmq_consume_creates_named_span():
    consumer = AgentRequestConsumer(
        connection_url='amqp://unused',
        queue_name='agent.requests',
        dlq_name='agent.requests.dlq',
        graph=_FakeGraph(),
        token_sink=_noop_sink,
    )

    class _FakeMessage:
        body = b'{"session_id": "s1", "message": "?"}'

        async def ack(self):
            return None

        async def nack(self, requeue: bool = True):
            return None

    await consumer._handle_message(_FakeMessage())

    assert 'rabbitmq.consume' in _span_names()
