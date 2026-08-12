import logging

from openinference.instrumentation import TraceConfig
from openinference.instrumentation.langchain import LangChainInstrumentor
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanExporter
from opentelemetry.util._once import Once

from app.core.settings import ObservabilitySettings

SERVICE_NAME: str = 'vera_agent_service'
logger = logging.getLogger(SERVICE_NAME)

_provider: TracerProvider | None = None
_shutdown = False


def _create_langchain_trace_config(
    trace_content_enabled: bool = True,
) -> TraceConfig:
    """Управляет экспортом содержимого LangChain/LangGraph spans.

    По умолчанию содержимое передаётся: без него в Phoenix не видно ни
    сообщений модели, ни промптов, ни результатов узлов графа. Явное
    отключение оставлено для окружений, где трейсы уходят наружу.
    """
    hide_content = not trace_content_enabled
    return TraceConfig(
        hide_inputs=hide_content,
        hide_outputs=hide_content,
        hide_input_messages=hide_content,
        hide_output_messages=hide_content,
        hide_input_text=hide_content,
        hide_output_text=hide_content,
        hide_prompts=hide_content,
        hide_choices=hide_content,
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
        config=_create_langchain_trace_config(settings.trace_content_enabled),
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
    `start_as_current_span` просто ничего не делает.

    Вызывать **в месте создания span**, а не сохранять результат в
    переменную модуля. До установки провайдера OpenTelemetry возвращает
    `ProxyTracer`, который при первом же span-е разрешается в реальный
    трейсер и навсегда запоминает тогдашний провайдер. Модульная переменная
    поэтому намертво привязывается к первому провайдеру процесса — в тестах
    это приводит к молчаливой потере spans после `reset_for_tests`.
    """
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


def _reset_otel_trace_globals() -> None:
    """Снимает однократную установку глобального `TracerProvider`.

    `opentelemetry.trace.set_tracer_provider()` защищён `Once`: повторный
    вызов в том же процессе молча игнорируется (лог-предупреждение
    "Overriding of current TracerProvider is not allowed"). Пока защита не
    снята, тест не может получить живой провайдер, если предыдущий тест уже
    установил свой и завершил его — например через lifespan приложения в
    `tests/integration/test_main.py`, который в `finally` вызывает
    `shutdown_tracing()`. Спаны после этого молча перестают записываться.

    Приватные атрибуты OpenTelemetry — единственный способ сбросить защиту
    без дополнительной зависимости `opentelemetry-test-utils`, которая
    существует ровно для этой цели и делает то же самое.
    """
    trace._TRACER_PROVIDER = None
    trace._TRACER_PROVIDER_SET_ONCE = Once()


def reset_for_tests(
    exporter: SpanExporter | None = None,
    trace_content_enabled: bool = False,
) -> TracerProvider:
    """Только для тестов. Настраивает провайдер с указанным экспортёром
    (например `InMemorySpanExporter`), чтобы проверить фактически
    созданные spans без реального Phoenix.

    Безопасно вызывать перед каждым тестом: сбрасывает и глобальное
    состояние OpenTelemetry, и подписку инструментации, поэтому тест
    получает заведомо живой провайдер независимо от того, что делали
    предыдущие тесты в этом процессе. На продуктовый `configure_tracing`
    не влияет — тот остаётся идемпотентным."""
    global _provider, _shutdown
    _provider = None
    _reset_otel_trace_globals()
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
        config=_create_langchain_trace_config(trace_content_enabled),
    )

    _provider = provider
    _shutdown = False
    return provider
