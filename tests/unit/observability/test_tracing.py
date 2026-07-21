import asyncio

import pytest
from langchain_mcp_adapters.interceptors import MCPToolCallRequest
from opentelemetry import propagate, trace
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

from app.clients.mcp_client import call_tool_with_retry, inject_trace_context
from app.core.settings import ObservabilitySettings
from app.messaging.consumer import AgentRequestConsumer
from app.observability.request_trace import get_request_trace
from app.observability.tracing import (
    _add_exporter,
    _create_langchain_trace_config,
    _create_otlp_exporter,
    force_flush_tracing,
    get_tracer,
    reset_for_tests,
    shutdown_tracing,
)
from app.streaming.session_bus import SessionBus

_exporter = InMemorySpanExporter()
reset_for_tests(_exporter)


@pytest.fixture(autouse=True)
def _clear_spans():
    _exporter.clear()
    yield


class _FakeTool:
    name = 'vera_rag_kb'

    def __init__(self, chunks: list[dict] | None = None):
        self._chunks = chunks or []

    async def ainvoke(self, arguments: dict):
        return {'chunks': self._chunks}


class _FlakyTool:
    name = 'vera_rag_kb'

    def __init__(self):
        self.call_count = 0

    async def ainvoke(self, arguments: dict):
        self.call_count += 1
        if self.call_count == 1:
            raise RuntimeError('temporary MCP failure')
        return {'chunks': []}


class _FakeGraph:
    def __init__(self, events: list[dict] | None = None, route: str = 'direct'):
        self._events = events or []
        self._route = route

    def astream_events(self, state, config, version='v2'):
        async def _generator():
            trace_data = get_request_trace()
            trace_data.route = self._route
            trace_data.search_required = self._route == 'knowledge_base'
            for event in self._events:
                yield event

        return _generator()


class _ToolCallingGraph:
    def __init__(self, tool=None):
        self._tool = tool or _FakeTool([{'chunk_id': 'c1'}])

    def astream_events(self, state, config, version='v2'):
        async def _generator():
            trace_data = get_request_trace()
            trace_data.route = 'knowledge_base'
            trace_data.search_required = True
            trace_data.tool_call_count = 1
            result = await call_tool_with_retry(
                self._tool,
                {'query': 'q'},
                retries=2,
                timeout_seconds=1.0,
            )
            trace_data.search_chunk_count = len(result['chunks'])
            yield {'event': 'on_chat_model_stream', 'data': {'chunk': _Chunk('Ответ')}}

        return _generator()


class _SequenceGraph:
    def __init__(self, events_per_call: list[list[dict | Exception]]):
        self._events_per_call = events_per_call
        self.call_count = 0

    def astream_events(self, state, config, version='v2'):
        events = self._events_per_call[self.call_count]
        self.call_count += 1

        async def _generator():
            get_request_trace().route = 'direct'
            for event in events:
                if isinstance(event, Exception):
                    raise event
                yield event

        return _generator()


class _DegradedGraph:
    def astream_events(self, state, config, version='v2'):
        async def _generator():
            trace_data = get_request_trace()
            trace_data.route = 'knowledge_base'
            trace_data.search_required = True
            trace_data.search_unavailable = True
            trace_data.tool_call_count = 1
            trace_data.outcome = 'degraded'
            yield {'event': 'on_chat_model_stream', 'data': {'chunk': _Chunk('fallback')}}

        return _generator()


class _Chunk:
    def __init__(self, content: str):
        self.content = content


class _FakeMessage:
    def __init__(self, body: bytes = b'{"session_id":"s1","message":"question"}'):
        self.body = body
        self.acked = False
        self.nacked = False

    async def ack(self):
        self.acked = True

    async def nack(self, requeue: bool = True):
        self.nacked = True


async def _noop_sink(session_id: str, event: dict) -> None:
    return None


def _finished_span(name: str):
    return next(span for span in _exporter.get_finished_spans() if span.name == name)


def _build_consumer(graph, capture: bool = False, max_chars: int = 12_000):
    return AgentRequestConsumer(
        connection_url='amqp://unused',
        queue_name='agent.requests',
        dlq_name='agent.requests.dlq',
        graph=graph,
        token_sink=_noop_sink,
        observability_settings=ObservabilitySettings(
            phoenix_capture_content=capture,
            phoenix_content_max_chars=max_chars,
        ),
    )


async def test_tool_call_creates_logical_span_with_aggregates():
    await call_tool_with_retry(_FakeTool(), {'query': 'q'}, retries=1, timeout_seconds=1.0)

    span = _finished_span('tool.vera_rag_kb')
    assert span.attributes['openinference.span.kind'] == 'TOOL'
    assert span.attributes['tool.retry.count'] == 0
    assert span.attributes['tool.outcome'] == 'empty'


async def test_publish_many_tokens_does_not_create_sse_spans():
    bus = SessionBus()
    for index in range(10):
        await bus.publish('s1', {'type': 'token', 'content': str(index)})

    assert not [span for span in _exporter.get_finished_spans() if span.name == 'sse.deliver']


async def test_message_creates_one_agent_root_with_safe_direct_attributes():
    graph = _FakeGraph(
        [
            {'event': 'on_chat_model_stream', 'data': {'chunk': _Chunk('A')}},
            {'event': 'on_chat_model_stream', 'data': {'chunk': _Chunk('Б')}},
        ]
    )
    consumer = _build_consumer(graph, capture=False)

    await consumer._handle_message(_FakeMessage())

    roots = [span for span in _exporter.get_finished_spans() if span.name == 'vera.agent.request']
    assert len(roots) == 1
    span = roots[0]
    assert span.parent is None
    assert span.attributes['agent.route'] == 'direct'
    assert span.attributes['agent.tool_call_count'] == 0
    assert span.attributes['agent.response.chunk_count'] == 2
    assert span.attributes['agent.response.char_count'] == 2
    assert span.attributes['agent.outcome'] == 'done'
    assert 'input.value' not in span.attributes
    assert 'output.value' not in span.attributes


async def test_capture_contains_only_current_message_and_truncated_final_output():
    graph = _FakeGraph(
        [{'event': 'on_chat_model_stream', 'data': {'chunk': _Chunk('123456')}}]
    )
    consumer = _build_consumer(graph, capture=True, max_chars=5)

    await consumer._handle_message(
        _FakeMessage(b'{"session_id":"s1","message":"question","user_id":"u1"}')
    )

    span = _finished_span('vera.agent.request')
    assert span.attributes['input.value'] == 'quest'
    assert span.attributes['input.mime_type'] == 'text/plain'
    assert span.attributes['input.truncated'] is True
    assert span.attributes['output.value'] == '12345'
    assert span.attributes['output.mime_type'] == 'text/plain'
    assert span.attributes['output.truncated'] is True
    assert span.attributes['agent.input.char_count'] == 8
    assert span.attributes['agent.response.char_count'] == 6
    assert span.attributes['user.authenticated'] is True


async def test_tool_span_is_child_of_agent_root_in_same_trace():
    consumer = _build_consumer(_ToolCallingGraph())

    await consumer._handle_message(_FakeMessage())

    root = _finished_span('vera.agent.request')
    tool_span = _finished_span('tool.vera_rag_kb')
    assert tool_span.context.trace_id == root.context.trace_id
    assert tool_span.parent.span_id == root.context.span_id
    assert root.attributes['agent.search.chunk_count'] == 1


async def test_empty_search_is_success_and_tool_retries_reach_root(monkeypatch):
    async def _no_sleep(_seconds):
        return None

    monkeypatch.setattr('app.clients.mcp_client.asyncio.sleep', _no_sleep)
    tool = _FlakyTool()
    consumer = _build_consumer(_ToolCallingGraph(tool))

    await consumer._handle_message(_FakeMessage())

    root = _finished_span('vera.agent.request')
    tool_span = _finished_span('tool.vera_rag_kb')
    assert tool.call_count == 2
    assert root.attributes['agent.search.chunk_count'] == 0
    assert root.attributes['agent.mcp.retry_count'] == 1
    assert root.attributes['agent.outcome'] == 'done'
    assert root.status.status_code is StatusCode.UNSET
    assert tool_span.attributes['tool.retry.count'] == 1
    assert tool_span.attributes['tool.outcome'] == 'empty'


async def test_retry_before_streaming_resets_output_and_is_counted():
    graph = _SequenceGraph(
        [
            [
                {'event': 'on_chat_model_stream', 'data': {'chunk': _Chunk('')}},
                RuntimeError('before streaming'),
            ],
            [{'event': 'on_chat_model_stream', 'data': {'chunk': _Chunk('final')}}],
        ]
    )
    consumer = _build_consumer(graph, capture=True)

    await consumer._handle_message(_FakeMessage())

    root = _finished_span('vera.agent.request')
    assert graph.call_count == 2
    assert root.attributes['agent.retry.count'] == 1
    assert root.attributes['output.value'] == 'final'
    assert root.attributes['agent.response.chunk_count'] == 1
    assert root.status.status_code is StatusCode.UNSET


async def test_error_after_first_token_is_terminal_root_error_without_retry():
    graph = _SequenceGraph(
        [
            [
                {'event': 'on_chat_model_stream', 'data': {'chunk': _Chunk('partial')}},
                RuntimeError('stream interrupted'),
            ]
        ]
    )
    consumer = _build_consumer(graph, capture=True)

    message = _FakeMessage()
    await consumer._handle_message(message)

    root = _finished_span('vera.agent.request')
    assert graph.call_count == 1
    assert message.acked is True
    assert root.attributes['agent.outcome'] == 'error'
    assert root.attributes['agent.streaming.started'] is True
    assert root.attributes['output.value'] == 'partial'
    assert root.status.status_code is StatusCode.ERROR


async def test_mcp_degradation_is_not_root_error():
    consumer = _build_consumer(_DegradedGraph())

    await consumer._handle_message(_FakeMessage())

    root = _finished_span('vera.agent.request')
    assert root.attributes['agent.outcome'] == 'degraded'
    assert root.attributes['agent.search.unavailable'] is True
    assert root.status.status_code is StatusCode.UNSET


async def test_invalid_payload_is_visible_as_failed_root():
    consumer = _build_consumer(_FakeGraph())

    await consumer._handle_message(_FakeMessage(b'not-json'))

    root = _finished_span('vera.agent.request')
    assert root.attributes['agent.outcome'] == 'invalid_payload'
    assert root.status.status_code is StatusCode.ERROR


async def test_interceptor_preserves_headers_and_injects_isolated_parallel_contexts():
    captured: dict[str, tuple[str, str]] = {}
    tracer = get_tracer()

    async def worker(name: str) -> None:
        with tracer.start_as_current_span(name) as span:
            request = MCPToolCallRequest(
                name='vera_rag_kb',
                args={'query': name},
                server_name='vera-tools',
                headers={'authorization': name},
            )

            async def handler(intercepted_request):
                await asyncio.sleep(0)
                carrier = intercepted_request.headers
                remote_context = propagate.extract(carrier)
                remote_span = trace.get_current_span(remote_context).get_span_context()
                captured[name] = (carrier['authorization'], carrier['traceparent'])
                assert remote_span.trace_id == span.get_span_context().trace_id
                return {'ok': True}

            await inject_trace_context(request, handler)

    await asyncio.gather(worker('first'), worker('second'))

    assert captured['first'][0] == 'first'
    assert captured['second'][0] == 'second'
    assert captured['first'][1] != captured['second'][1]


def test_exporter_uses_phoenix_project_header(monkeypatch):
    calls = {}

    class _Exporter:
        def __init__(self, **kwargs):
            calls.update(kwargs)

    monkeypatch.setattr('app.observability.tracing.OTLPSpanExporter', _Exporter)

    _create_otlp_exporter(
        ObservabilitySettings(
            phoenix_otlp_endpoint='http://phoenix:6006/v1/traces',
            phoenix_project_name='vera-testing',
        )
    )

    assert calls == {
        'endpoint': 'http://phoenix:6006/v1/traces',
        'headers': {'x-project-name': 'vera-testing'},
    }


def test_langchain_auto_spans_never_capture_conversation_content():
    config = _create_langchain_trace_config()

    assert config.hide_inputs is True
    assert config.hide_outputs is True
    assert config.hide_input_messages is True
    assert config.hide_output_messages is True
    assert config.hide_prompts is True
    assert config.hide_choices is True


def test_disabled_phoenix_does_not_create_exporter(monkeypatch):
    class _Provider:
        def add_span_processor(self, processor):
            raise AssertionError('processor must not be added')

    monkeypatch.setattr(
        'app.observability.tracing._create_otlp_exporter',
        lambda settings: (_ for _ in ()).throw(AssertionError('exporter must not be created')),
    )

    _add_exporter(_Provider(), ObservabilitySettings(phoenix_enabled=False))


def test_shutdown_force_flushes_and_closes_provider_once(monkeypatch):
    calls: list[tuple[str, int | None]] = []

    class _Provider:
        def force_flush(self, timeout_millis: int):
            calls.append(('flush', timeout_millis))
            return True

        def shutdown(self):
            calls.append(('shutdown', None))

    monkeypatch.setattr('app.observability.tracing._provider', _Provider())
    monkeypatch.setattr('app.observability.tracing._shutdown', False)

    shutdown_tracing()
    shutdown_tracing()

    assert calls == [('flush', 10_000), ('shutdown', None)]


def test_force_flush_failure_is_soft(monkeypatch):
    class _Provider:
        def force_flush(self, timeout_millis: int):
            raise RuntimeError('Phoenix unavailable')

    monkeypatch.setattr('app.observability.tracing._provider', _Provider())
    monkeypatch.setattr('app.observability.tracing._shutdown', False)

    assert force_flush_tracing() is False
