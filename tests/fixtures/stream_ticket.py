import base64
import hashlib
import hmac
import json


def create_stream_ticket(
    *,
    api_key: str,
    request_id: str,
    session_id: str = 'session-1',
    user_id: str | None = 'user@example.com',
    anonymous_token_hash: str | None = None,
    expires_at: int,
) -> str:
    """Создаёт ticket независимо от production verifier."""
    payload: dict[str, object] = {
        'exp': expires_at,
        'request_id': request_id,
        'session_id': session_id,
    }
    if user_id is not None:
        payload['user_id'] = user_id
    if anonymous_token_hash is not None:
        payload['anonymous_token_hash'] = anonymous_token_hash

    encoded_payload = (
        base64.urlsafe_b64encode(
            json.dumps(
                payload,
                separators=(',', ':'),
                sort_keys=True,
            ).encode()
        )
        .decode()
        .rstrip('=')
    )
    derived_key = hmac.new(
        api_key.encode(),
        b'vera-stream-ticket',
        hashlib.sha256,
    ).digest()
    signature = hmac.new(
        derived_key,
        encoded_payload.encode(),
        hashlib.sha256,
    ).digest()
    encoded_signature = base64.urlsafe_b64encode(signature).decode().rstrip('=')
    return f'{encoded_payload}.{encoded_signature}'
