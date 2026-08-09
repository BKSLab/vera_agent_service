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
