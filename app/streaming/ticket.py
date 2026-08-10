import base64
import binascii
import hashlib
import hmac
import json
import time
from dataclasses import dataclass

from app.exceptions.streaming import InvalidStreamTicketError

STREAM_TICKET_CONTEXT = b'vera-stream-ticket'


@dataclass(frozen=True)
class StreamTicketClaims:
    """Проверенные данные владельца и запроса из stream ticket."""

    request_id: str
    session_id: str
    user_id: str | None
    anonymous_token_hash: str | None
    expires_at: int


class StreamTicketVerifier:
    """Проверяет короткоживущие HMAC-ticket для подключения к SSE."""

    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError('API key для проверки stream ticket не задан')
        self._signing_key = hmac.new(
            api_key.encode(),
            STREAM_TICKET_CONTEXT,
            hashlib.sha256,
        ).digest()

    def verify(
        self,
        ticket: str | None,
        *,
        request_id: str,
        now: int | None = None,
    ) -> StreamTicketClaims:
        """Проверяет подпись, срок и привязку ticket к request_id.

        Args:
            ticket: Ticket из query-параметра SSE-запроса.
            request_id: Идентификатор запроса из URL.
            now: Текущее Unix-время для детерминированных тестов.

        Returns:
            Проверенные claims ticket.

        Raises:
            InvalidStreamTicketError: Ticket отсутствует либо не прошёл проверку.
        """
        if not ticket:
            raise InvalidStreamTicketError

        try:
            encoded_payload, encoded_signature = ticket.split('.', 1)
            provided_signature = _decode_base64url(encoded_signature)
            expected_signature = hmac.new(
                self._signing_key,
                encoded_payload.encode(),
                hashlib.sha256,
            ).digest()
            if not hmac.compare_digest(provided_signature, expected_signature):
                raise InvalidStreamTicketError

            payload = json.loads(_decode_base64url(encoded_payload))
            claims = _parse_claims(payload)
        except (
            ValueError,
            TypeError,
            UnicodeDecodeError,
            json.JSONDecodeError,
            binascii.Error,
        ) as error:
            raise InvalidStreamTicketError from error

        current_time = int(time.time()) if now is None else now
        if claims.request_id != request_id or claims.expires_at <= current_time:
            raise InvalidStreamTicketError
        return claims


def _decode_base64url(value: str) -> bytes:
    return base64.b64decode(
        value + '=' * (-len(value) % 4),
        altchars=b'-_',
        validate=True,
    )


def _parse_claims(payload: object) -> StreamTicketClaims:
    if not isinstance(payload, dict):
        raise InvalidStreamTicketError

    request_id = payload.get('request_id')
    session_id = payload.get('session_id')
    user_id = payload.get('user_id')
    anonymous_token_hash = payload.get('anonymous_token_hash')
    expires_at = payload.get('exp')
    has_user_id = user_id is not None
    has_anonymous_owner = anonymous_token_hash is not None
    if (
        not isinstance(request_id, str)
        or not request_id
        or not isinstance(session_id, str)
        or not session_id
        or has_user_id == has_anonymous_owner
        or (has_user_id and (not isinstance(user_id, str) or not user_id))
        or (has_anonymous_owner and (not isinstance(anonymous_token_hash, str) or not anonymous_token_hash))
        or type(expires_at) is not int
    ):
        raise InvalidStreamTicketError

    return StreamTicketClaims(
        request_id=request_id,
        session_id=session_id,
        user_id=user_id,
        anonymous_token_hash=anonymous_token_hash,
        expires_at=expires_at,
    )
