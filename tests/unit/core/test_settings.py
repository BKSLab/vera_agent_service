import pytest
from pydantic import ValidationError

from app.core.settings import (
    DBSettings,
    LlmSettings,
    McpSettings,
    RabbitMQSettings,
    RedisSettings,
    Settings,
    StreamingSettings,
)


@pytest.mark.parametrize(
    ('settings', 'expected_url'),
    [
        (
            DBSettings(
                postgres_host='db.local',
                postgres_port=5432,
                postgres_user='user:name',
                postgres_password='p@ss/word',
                postgres_name='db/name',
            ),
            'postgresql+asyncpg://user%3Aname:p%40ss%2Fword@db.local:5432/db%2Fname',
        ),
        (
            RabbitMQSettings(
                rabbitmq_host='rabbit.local',
                rabbitmq_port=5672,
                rabbitmq_user='rabbit/user',
                rabbitmq_password='p@ss:word',
                rabbitmq_vhost='/vera vhost',
            ),
            'amqp://rabbit%2Fuser:p%40ss%3Aword@rabbit.local:5672/%2Fvera%20vhost',
        ),
        (
            RedisSettings(
                redis_host='redis.local',
                redis_port=6379,
                redis_password='p@ss/word',
                redis_db=2,
            ),
            'redis://:p%40ss%2Fword@redis.local:6379/2',
        ),
    ],
)
def test_connection_url_percent_encodes_reserved_components(
    settings: DBSettings | RabbitMQSettings | RedisSettings,
    expected_url: str,
) -> None:
    assert settings.url_connect == expected_url


def test_streaming_deadline_covers_longest_tool_timeout() -> None:
    assert StreamingSettings().sse_request_deadline_seconds >= 360


def test_redis_session_ttl_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        RedisSettings(
            redis_host='redis.local',
            redis_port=6379,
            redis_session_ttl_seconds=0,
        )


def test_streaming_deadline_rejects_value_shorter_than_longest_tool() -> None:
    with pytest.raises(ValidationError):
        StreamingSettings(sse_request_deadline_seconds=359)


def test_late_buffer_must_fit_subscriber_queue() -> None:
    with pytest.raises(ValidationError, match='late buffer'):
        StreamingSettings(
            sse_subscriber_queue_max_events=1,
            sse_late_buffer_max_events=2,
        )


def test_late_buffer_request_limit_must_fit_total_state_limit() -> None:
    with pytest.raises(ValidationError, match='общего лимита state'):
        StreamingSettings(
            sse_late_buffer_max_requests=2,
            sse_request_state_max_entries=1,
        )


def test_streaming_deadline_uses_configured_email_tool_timeout() -> None:
    settings = Settings.model_construct(
        streaming=StreamingSettings(sse_request_deadline_seconds=420),
        mcp=McpSettings(mcp_consultation_email_timeout_seconds=421),
    )

    with pytest.raises(ValueError, match='consultation email'):
        settings.validate_stream_deadline_covers_longest_tool()


def test_llm_defaults_to_gemini_3_7_flash() -> None:
    assert LlmSettings.model_fields['llm_model'].default == 'google/gemini-3.7-flash'


def test_llm_temperature_defaults_to_0_3() -> None:
    settings = LlmSettings(
        llm_api_key='test-key',
        llm_api_url='http://mock/v1',
        _env_file=None,
    )

    assert settings.llm_temperature == 0.3
    assert settings.llm_reasoning_effort is None
