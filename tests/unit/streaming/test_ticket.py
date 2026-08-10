import pytest

from app.exceptions.streaming import InvalidStreamTicketError
from app.streaming.ticket import StreamTicketVerifier
from tests.fixtures.stream_ticket import create_stream_ticket

API_KEY = 'shared-test-key'
NOW = 1_700_000_000
GOLDEN_TICKET = (
    'eyJleHAiOjE3MDAwMDAwNjAsInJlcXVlc3RfaWQiOiJyZXF1ZXN0LTEiLCJzZXNzaW9uX2lk'
    'Ijoic2Vzc2lvbi0xIiwidXNlcl9pZCI6InVzZXJAZXhhbXBsZS5jb20ifQ.'
    'e7TqCWduSK_4Rqwqus697QWRJMt-4JTdjtcVrPFEoEA'
)


def test_verifier_accepts_user_ticket_and_returns_bound_claims():
    ticket = create_stream_ticket(
        api_key=API_KEY,
        request_id='request-1',
        expires_at=NOW + 60,
    )

    assert ticket == GOLDEN_TICKET

    claims = StreamTicketVerifier(API_KEY).verify(
        ticket,
        request_id='request-1',
        now=NOW,
    )

    assert claims.request_id == 'request-1'
    assert claims.session_id == 'session-1'
    assert claims.user_id == 'user@example.com'
    assert claims.anonymous_token_hash is None
    assert claims.expires_at == NOW + 60


def test_verifier_accepts_anonymous_owner_ticket():
    ticket = create_stream_ticket(
        api_key=API_KEY,
        request_id='request-1',
        user_id=None,
        anonymous_token_hash='a' * 64,
        expires_at=NOW + 60,
    )

    claims = StreamTicketVerifier(API_KEY).verify(
        ticket,
        request_id='request-1',
        now=NOW,
    )

    assert claims.user_id is None
    assert claims.anonymous_token_hash == 'a' * 64


@pytest.mark.parametrize(
    'ticket',
    [
        None,
        '',
        'not-a-ticket',
        create_stream_ticket(
            api_key='wrong-key',
            request_id='request-1',
            expires_at=NOW + 60,
        ),
        create_stream_ticket(
            api_key=API_KEY,
            request_id='request-1',
            expires_at=NOW,
        ),
        create_stream_ticket(
            api_key=API_KEY,
            request_id='request-1',
            user_id=None,
            expires_at=NOW + 60,
        ),
        create_stream_ticket(
            api_key=API_KEY,
            request_id='request-1',
            anonymous_token_hash='a' * 64,
            expires_at=NOW + 60,
        ),
    ],
)
def test_verifier_rejects_invalid_ticket(ticket):
    with pytest.raises(InvalidStreamTicketError):
        StreamTicketVerifier(API_KEY).verify(
            ticket,
            request_id='request-1',
            now=NOW,
        )


def test_verifier_rejects_ticket_for_another_request_id():
    ticket = create_stream_ticket(
        api_key=API_KEY,
        request_id='request-1',
        expires_at=NOW + 60,
    )

    with pytest.raises(InvalidStreamTicketError):
        StreamTicketVerifier(API_KEY).verify(
            ticket,
            request_id='request-2',
            now=NOW,
        )
