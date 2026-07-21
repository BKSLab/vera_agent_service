import logging

from openinference.instrumentation import TraceConfig
from openinference.instrumentation.langchain import LangChainInstrumentor
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanExporter

from app.core.settings import ObservabilitySettings

SERVICE_NAME: str = 'vera_agent_service'
logger = logging.getLogger(SERVICE_NAME)

_provider: TracerProvider | None = None
_shutdown = False


def _create_langchain_trace_config() -> TraceConfig:
    """Не позволяет автоинструментации экспортировать содержимое диалога.

    `PHOENIX_CAPTURE_CONTENT` относится только к ручному корневому span
    `vera.agent.request`: там текст нового вопроса и финального ответа
    обрезается и добавляется осознанно. LangChain spans видят system prompt,
    полную историю и внутренние ToolMessage, поэтому их input/output скрыты
    всегда — в том числе при разрешённом capture на корне.
    """
    return TraceConfig(
        hide_inputs=True,
        hide_outputs=True,
        hide_input_messages=True,
        hide_output_messages=True,
        hide_prompts=True,
        hide_choices=True,
    )


def configure_tracing(settings: ObservabilitySettings) -> TracerProvider:
    """Инициализирует OpenTelemetry + `openinference-instrumentation-langchain`
    (Этап 9, AGENT_VERA_ARCHITECTURE.md раздел "Observability").

    `LangChainInstrumentor` автоматически создаёт spans для всех вызовов
    LangChain/LangGraph внутри процесса (chat-модель, узлы графа) — не
    нужно расставлять их вручную в `app/graph/nodes/*`. Вручную остаются
    только продуктовый root `vera.agent.request` и один логический
    `tool.<name>` на границе MCP; доставка отдельных SSE-токенов spans не создаёт.

    Идемпотентна — повторный вызов (например в тестах) возвращает уже
    созданный `TracerProvider`, не плодит дублирующиеся
    процессоры/подписки на инструментацию.
    """
    global _provider, _shutdown
    if _provider is not None:
        return _provider

    provider = TracerProvider(resource=Resource.create({'service.name': SERVICE_NAME}))
    _add_exporter(provider, settings)
    trace.set_tracer_provider(provider)
    LangChainInstrumentor().instrument(
        tracer_provider=provider,
        config=_create_langchain_trace_config(),
    )

    _provider = provider
    _shutdown = False
    return provider


def _create_otlp_exporter(settings: ObservabilitySettings) -> OTLPSpanExporter:
    return OTLPSpanExporter(
        endpoint=settings.phoenix_otlp_endpoint,
        headers={'x-project-name': settings.phoenix_project_name},
    )


def _add_exporter(provider: TracerProvider, settings: ObservabilitySettings) -> None:
    if settings.phoenix_enabled:
        provider.add_span_processor(BatchSpanProcessor(_create_otlp_exporter(settings)))


def get_tracer() -> trace.Tracer:
    """Трейсер для ручных spans. Безопасен для вызова до
    `configure_tracing()` (например в юнит-тестах, не поднимающих Phoenix)
    — без настроенного провайдера OpenTelemetry отдаёт no-op трейсер,
    `start_as_current_span` просто ничего не делает."""
    return trace.get_tracer(SERVICE_NAME)


def force_flush_tracing(timeout_millis: int = 10_000) -> bool:
    """Доставляет завершённые spans, не влияя на остановку приложения при ошибке exporter."""
    if _provider is None or _shutdown:
        return True
    try:
        return _provider.force_flush(timeout_millis=timeout_millis)
    except Exception:  # noqa: BLE001 - observability не должна ломать lifecycle сервиса
        logger.exception('Не удалось выполнить force_flush OpenTelemetry')
        return False


def shutdown_tracing(timeout_millis: int = 10_000) -> None:
    """Идемпотентно flush-ит и завершает настроенный provider."""
    global _shutdown
    if _provider is None or _shutdown:
        return
    force_flush_tracing(timeout_millis=timeout_millis)
    try:
        _provider.shutdown()
    except Exception:  # noqa: BLE001 - observability не должна ломать остановку сервиса
        logger.exception('Не удалось завершить OpenTelemetry provider')
    finally:
        _shutdown = True


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
    global _provider, _shutdown
    _provider = None
    provider = TracerProvider(resource=Resource.create({'service.name': SERVICE_NAME}))
    if exporter is not None:
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor

        provider.add_span_processor(SimpleSpanProcessor(exporter))
    trace.set_tracer_provider(provider)

    instrumentor = LangChainInstrumentor()
    if instrumentor.is_instrumented_by_opentelemetry:
        instrumentor.uninstrument()
    instrumentor.instrument(
        tracer_provider=provider,
        config=_create_langchain_trace_config(),
    )

    _provider = provider
    _shutdown = False
    return provider
