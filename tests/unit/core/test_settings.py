import pytest

from app.core.settings import DBSettings, RabbitMQSettings, RedisSettings


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
