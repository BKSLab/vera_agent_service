from openinference.instrumentation.langchain import LangChainInstrumentor
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanExporter

from app.core.settings import ObservabilitySettings

SERVICE_NAME: str = 'vera_agent_service'

_provider: TracerProvider | None = None


def configure_tracing(settings: ObservabilitySettings) -> TracerProvider:
    """Инициализирует OpenTelemetry + `openinference-instrumentation-langchain`
    (Этап 9, AGENT_VERA_ARCHITECTURE.md раздел "Observability").

    `LangChainInstrumentor` автоматически создаёт spans для всех вызовов
    LangChain/LangGraph внутри процесса (chat-модель, узлы графа) — не
    нужно расставлять их вручную в `app/graph/nodes/*`. Ручные spans
    (`rabbitmq.consume`, `mcp.tool_call`, `sse.deliver`) добавлены в
    `app/messaging/consumer.py`, `app/clients/mcp_client.py`,
    `app/streaming/session_bus.py` — на границах, которые эта
    автоинструментация не покрывает.

    Идемпотентна — повторный вызов (например в тестах) возвращает уже
    созданный `TracerProvider`, не плодит дублирующиеся
    процессоры/подписки на инструментацию.
    """
    global _provider
    if _provider is not None:
        return _provider

    provider = TracerProvider(resource=Resource.create({'service.name': SERVICE_NAME}))
    exporter = OTLPSpanExporter(endpoint=settings.phoenix_otlp_endpoint)
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    LangChainInstrumentor().instrument(tracer_provider=provider)

    _provider = provider
    return provider


def get_tracer() -> trace.Tracer:
    """Трейсер для ручных spans. Безопасен для вызова до
    `configure_tracing()` (например в юнит-тестах, не поднимающих Phoenix)
    — без настроенного провайдера OpenTelemetry отдаёт no-op трейсер,
    `start_as_current_span` просто ничего не делает."""
    return trace.get_tracer(SERVICE_NAME)


def reset_for_tests(exporter: SpanExporter | None = None) -> TracerProvider:
    """Только для тестов. Настраивает провайдер с указанным экспортёром
    (например `InMemorySpanExporter`), чтобы проверить фактически
    созданные spans без реального Phoenix.

    **Вызывать один раз за тестовый процесс** (например на уровне модуля
    теста), не в каждом тесте: `opentelemetry.trace.set_tracer_provider()`
    можно успешно вызвать только один раз за процесс — повторные вызовы
    молча игнорируются самим OpenTelemetry SDK (лог-предупреждение
    "Overriding of current TracerProvider is not allowed", найдено
    эмпирически при реализации Этапа 9). Между тестами достаточно
    `exporter.clear()` — сам провайдер и подписка на инструментацию
    переиспользуются."""
    global _provider
    _provider = None
    provider = TracerProvider(resource=Resource.create({'service.name': SERVICE_NAME}))
    if exporter is not None:
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor

        provider.add_span_processor(SimpleSpanProcessor(exporter))
    trace.set_tracer_provider(provider)

    instrumentor = LangChainInstrumentor()
    if instrumentor.is_instrumented_by_opentelemetry:
        instrumentor.uninstrument()
    instrumentor.instrument(tracer_provider=provider)

    _provider = provider
    return provider
