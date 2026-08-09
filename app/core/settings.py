from functools import lru_cache
from urllib.parse import quote

from pydantic import Field, SecretStr
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
    redis_session_ttl_seconds: int = 86400
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
    llm_model: str
    llm_temperature: float = 0.3


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


class ObservabilitySettings(SettingsBase):
    """Настройки экспорта трейсов в Arize Phoenix (Этап 9, AGENT_SERVICE_PLAN.md)."""

    phoenix_enabled: bool = True
    phoenix_otlp_endpoint: str = 'http://localhost:6006/v1/traces'
    phoenix_project_name: str = 'vera-local'
    trace_content_enabled: bool = False
    """Явный opt-in на экспорт полного содержимого диалогов и LLM-вызовов."""


class Settings(BaseSettings):
    """Агрегатор всех доменных настроек проекта."""

    app: AppSettings = Field(default_factory=AppSettings)
    db: DBSettings = Field(default_factory=DBSettings)
    rabbitmq: RabbitMQSettings = Field(default_factory=RabbitMQSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    llm: LlmSettings = Field(default_factory=LlmSettings)
    mcp: McpSettings = Field(default_factory=McpSettings)
    observability: ObservabilitySettings = Field(default_factory=ObservabilitySettings)


@lru_cache
def get_settings() -> Settings:
    return Settings()
