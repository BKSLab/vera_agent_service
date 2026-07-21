from functools import lru_cache

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

    @property
    def url_connect(self) -> str:
        return (
            f'amqp://{self.rabbitmq_user}:'
            f'{self.rabbitmq_password.get_secret_value()}@'
            f'{self.rabbitmq_host}:{self.rabbitmq_port}/'
            f'{self.rabbitmq_vhost}'
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
        auth = f':{self.redis_password.get_secret_value()}@' if self.redis_password else ''
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
    """Retry на уровне MCP-клиента — независим от retry-политики самого
    RabbitMQ-сообщения (Этап 3.2/6.3 плана, раздел 6, открытый вопрос:
    конкретные значения таймаута/числа повторов подлежат подтверждению)."""


class ObservabilitySettings(SettingsBase):
    """Настройки экспорта трейсов в Arize Phoenix (Этап 9, AGENT_SERVICE_PLAN.md)."""

    phoenix_enabled: bool = True
    phoenix_otlp_endpoint: str = 'http://localhost:6006/v1/traces'
    phoenix_project_name: str = 'vera-local'
    phoenix_capture_content: bool = False
    phoenix_content_max_chars: int = Field(default=12_000, ge=1)


class Settings(BaseSettings):
    """Агрегатор всех доменных настроек проекта."""

    app: AppSettings = Field(default_factory=AppSettings)
    rabbitmq: RabbitMQSettings = Field(default_factory=RabbitMQSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    llm: LlmSettings = Field(default_factory=LlmSettings)
    mcp: McpSettings = Field(default_factory=McpSettings)
    observability: ObservabilitySettings = Field(default_factory=ObservabilitySettings)


@lru_cache
def get_settings() -> Settings:
    return Settings()
