import pytest
from pydantic import ValidationError

from app.messaging.schemas import AgentRequestMessage


@pytest.mark.parametrize('user_id_length', [99, 100, 101, 255])
def test_agent_request_message_accepts_supported_user_id_lengths(
    user_id_length: int,
) -> None:
    payload = AgentRequestMessage(
        session_id='session-1',
        request_id='request-1',
        user_id='u' * user_id_length,
        message='Вопрос',
    )

    assert len(payload.user_id or '') == user_id_length


def test_agent_request_message_rejects_user_id_longer_than_255() -> None:
    with pytest.raises(ValidationError):
        AgentRequestMessage(
            session_id='session-1',
            request_id='request-1',
            user_id='u' * 256,
            message='Вопрос',
        )


@pytest.mark.parametrize('session_id_length', [1, 100])
def test_agent_request_message_accepts_supported_session_id_lengths(
    session_id_length: int,
) -> None:
    payload = AgentRequestMessage(
        session_id='s' * session_id_length,
        request_id='request-1',
        user_id='user-1',
        message='Вопрос',
    )

    assert len(payload.session_id) == session_id_length


def test_agent_request_message_rejects_session_id_longer_than_100() -> None:
    with pytest.raises(ValidationError):
        AgentRequestMessage(
            session_id='s' * 101,
            request_id='request-1',
            user_id='user-1',
            message='Вопрос',
        )


def test_agent_request_message_rejects_missing_owner() -> None:
    with pytest.raises(ValidationError, match='Должен быть указан владелец запроса'):
        AgentRequestMessage(
            session_id='session-1',
            request_id='request-1',
            message='Вопрос',
        )


def test_agent_request_message_rejects_empty_user_id() -> None:
    with pytest.raises(ValidationError):
        AgentRequestMessage(
            session_id='session-1',
            request_id='request-1',
            user_id='',
            message='Вопрос',
        )


def test_agent_request_message_accepts_anonymous_owner() -> None:
    payload = AgentRequestMessage(
        session_id='session-1',
        request_id='request-1',
        anonymous_token_hash='a' * 64,
        message='Вопрос',
    )

    assert payload.user_id is None
    assert payload.anonymous_token_hash == 'a' * 64


def test_agent_request_message_accepts_both_owner_mechanisms() -> None:
    payload = AgentRequestMessage(
        session_id='session-1',
        request_id='request-1',
        user_id='user-1',
        anonymous_token_hash='a' * 64,
        message='Вопрос',
    )

    assert payload.user_id == 'user-1'
    assert payload.anonymous_token_hash == 'a' * 64
