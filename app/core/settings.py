from functools import lru_cache
from typing import Literal
from urllib.parse import quote

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class SettingsBase(BaseSettings):
    """Базовый класс для всех доменных настроек проекта."""

    model_config = SettingsConfigDict(
        env_file='.env',
        env_file_encoding='utf-8',
        extra='ignore',
    )


class AppSettings(SettingsBase):
    """Общие настройки приложения."""

    app_name: str = 'vera_agent_service'
    logging_config_path: str = 'logging.ini'
    secret_key: SecretStr
    admin_login: str
    admin_password: SecretStr
    admin_session_https_only: bool = False
    api_key: SecretStr


class StreamingSettings(SettingsBase):
    """Ограничения in-memory SSE transport одного процесса."""

    sse_subscriber_queue_max_events: int = Field(default=256, ge=1)
    sse_late_buffer_max_events: int = Field(default=256, ge=1)
    sse_late_buffer_max_requests: int = Field(default=1024, ge=1)
    sse_request_state_max_entries: int = Field(default=2048, ge=1)
    sse_late_buffer_ttl_seconds: float = Field(default=60.0, gt=0)
    sse_buffer_cleanup_interval_seconds: float = Field(default=15.0, gt=0)
    sse_heartbeat_interval_seconds: float = Field(default=15.0, gt=0)
    sse_request_deadline_seconds: float = Field(default=420.0, ge=360.0)
    """Deadline больше 360-секундного timeout consultation email tool."""

    @model_validator(mode='after')
    def validate_buffer_fits_subscriber_queue(self) -> 'StreamingSettings':
        if (
            self.sse_late_buffer_max_events
            > self.sse_subscriber_queue_max_events
        ):
            raise ValueError(
                'SSE late buffer не может быть больше subscriber queue'
            )
        if (
            self.sse_late_buffer_max_requests
            > self.sse_request_state_max_entries
        ):
            raise ValueError(
                'Лимит SSE late buffers не может быть больше общего лимита state'
            )
        return self


class DBSettings(SettingsBase):
    """Настройки подключения к PostgreSQL."""

    postgres_host: str
    postgres_port: int
    postgres_user: str
    postgres_password: SecretStr
    postgres_name: str

    @property
    def url_connect(self) -> str:
        """Формирует строку асинхронного подключения SQLAlchemy."""
        user = quote(self.postgres_user, safe='')
        password = quote(self.postgres_password.get_secret_value(), safe='')
        database = quote(self.postgres_name, safe='')
        return (
            f'postgresql+asyncpg://{user}:'
            f'{password}@'
            f'{self.postgres_host}:{self.postgres_port}/'
            f'{database}'
        )


class RabbitMQSettings(SettingsBase):
    """Настройки подключения к RabbitMQ — очередь `agent.requests` (Этап 6,
    AGENT_SERVICE_PLAN.md)."""

    rabbitmq_host: str
    rabbitmq_port: int
    rabbitmq_user: str
    rabbitmq_password: SecretStr
    rabbitmq_vhost: str = '/'
    rabbitmq_queue: str = 'agent.requests'
    rabbitmq_dlq: str = 'agent.requests.dlq'

    turn_lease_seconds: float = 900.0
    """Срок аренды обрабатываемой реплики.

    Пока аренда жива, повторная доставка того же `request_id` считается
    настоящим дубликатом; после истечения реплику можно безопасно
    перезахватить. Значение обязано с запасом превышать самый долгий
    инструмент (`mcp_consultation_email_timeout_seconds`, 360 секунд).
    """

    turn_stale_after_seconds: float = 1800.0
    """Через сколько после истечения аренды брошенная реплика закрывается.

    Запас относительно `turn_lease_seconds` нужен, чтобы startup-очистка не
    обгоняла штатную повторную доставку сразу после рестарта: иначе живой
    запрос был бы закрыт раньше, чем брокер вернёт его сообщение.
    """

    @property
    def url_connect(self) -> str:
        user = quote(self.rabbitmq_user, safe='')
        password = quote(self.rabbitmq_password.get_secret_value(), safe='')
        vhost = quote(self.rabbitmq_vhost, safe='')
        return (
            f'amqp://{user}:'
            f'{password}@'
            f'{self.rabbitmq_host}:{self.rabbitmq_port}/'
            f'{vhost}'
        )


class RedisSettings(SettingsBase):
    """Настройки подключения к Redis — LangGraph checkpointer (Этап 5,
    AGENT_SERVICE_PLAN.md)."""

    redis_host: str
    redis_port: int
    redis_password: SecretStr | None = None
    redis_db: int = 0
    redis_session_ttl_seconds: int = Field(default=86400, gt=0)
    """TTL ключей сессии в Redis. 86400с (24 часа неактивности) — предложенное
    по умолчанию значение (AGENT_SERVICE_PLAN.md, раздел 6, открытый вопрос),
    подлежит подтверждению перед реализацией Этапа 5."""

    @property
    def url_connect(self) -> str:
        auth = (
            f':{quote(self.redis_password.get_secret_value(), safe="")}@'
            if self.redis_password
            else ''
        )
        return f'redis://{auth}{self.redis_host}:{self.redis_port}/{self.redis_db}'


class LlmSettings(SettingsBase):
    """Настройки доступа к LLM-провайдеру для графа (Этап 2).

    Провайдер конфигурируется (AGENT_VERA_ARCHITECTURE.md) — любой
    OpenAI-совместимый Chat Completions API, не захардкожен на конкретного
    поставщика.
    """

    llm_api_key: SecretStr
    llm_api_url: str
    llm_model: str = 'google/gemini-3.7-flash'
    llm_temperature: float | None = 0.3
    llm_reasoning_effort: Literal['low'] | None = None
    """Опциональный режим Polza `reasoning={"effort": "low"}`.

    По умолчанию не меняет payload. В production включается через
    `LLM_REASONING_EFFORT=low`, поэтому режим можно откатить без правки кода.
    """


class McpSettings(SettingsBase):
    """Настройки подключения к MCP Tools Server (Этап 3).

    В тестах MCP Tools Server подменяется локальным мок-сервером; в рабочем
    окружении клиент подключается к адресу из `MCP_SERVER_URL`.
    """

    mcp_server_url: str = 'http://localhost:9000/mcp'
    mcp_call_timeout_seconds: float = 5.0
    mcp_call_retries: int = 2
    """Retry только для безопасных идемпотентных MCP-вызовов."""

    mcp_consultation_email_timeout_seconds: float = 360.0
    """Отдельный timeout мутирующей тулы: внутри MCP выполняются LLM,
    PDF-рендер и SMTP. Внешние retries для неё запрещены."""


class GraphContextSettings(SettingsBase):
    """Бюджет контекста, передаваемого модели графом (VERA-020).

    История диалога продолжает накапливаться в Redis-checkpoint'е и в
    PostgreSQL без ограничения — эти настройки ограничивают только то,
    что уходит в конкретный вызов LLM (`app/graph/context_budget.py`). Новые
    пользовательские сообщения сохраняются в Redis уже обезличенными.
    """

    context_max_turns: int = Field(default=12, ge=1)
    """Сколько последних реплик (вопрос пользователя + ответ ассистента, с
    учётом промежуточных tool-сообщений) передаются модели полностью."""

    context_older_turns_summary_max_chars: int = Field(default=800, ge=0)
    """Максимальная длина безопасной текстовой выжимки одной более старой
    реплики в сводном `SystemMessage`. `0` — выжимка не обрезается."""


class ObservabilitySettings(SettingsBase):
    """Настройки экспорта трейсов в Arize Phoenix (Этап 9, AGENT_SERVICE_PLAN.md)."""

    phoenix_enabled: bool = True
    phoenix_otlp_endpoint: str = 'http://localhost:6006/v1/traces'
    phoenix_project_name: str = 'vera-local'
    trace_content_enabled: bool = True
    """Экспорт полного содержимого диалогов, промптов и вызовов инструментов.

    Включено: без содержимого Phoenix показывает только дерево спанов и
    тайминги, а разобрать по трейсу что пришло на вход, какой инструмент
    вызвался, с чем и что вернул — нельзя. Доступ к самому Phoenix
    ограничивается на уровне сервиса, а не вырезанием данных из трейсов.
    """


class Settings(BaseSettings):
    """Агрегатор всех доменных настроек проекта."""

    app: AppSettings = Field(default_factory=AppSettings)
    streaming: StreamingSettings = Field(default_factory=StreamingSettings)
    db: DBSettings = Field(default_factory=DBSettings)
    rabbitmq: RabbitMQSettings = Field(default_factory=RabbitMQSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    llm: LlmSettings = Field(default_factory=LlmSettings)
    mcp: McpSettings = Field(default_factory=McpSettings)
    graph_context: GraphContextSettings = Field(default_factory=GraphContextSettings)
    observability: ObservabilitySettings = Field(default_factory=ObservabilitySettings)

    @model_validator(mode='after')
    def validate_stream_deadline_covers_longest_tool(self) -> 'Settings':
        if (
            self.streaming.sse_request_deadline_seconds
            < self.mcp.mcp_consultation_email_timeout_seconds
        ):
            raise ValueError(
                'SSE request deadline не может быть короче timeout consultation email tool'
            )
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
