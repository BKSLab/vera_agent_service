from datetime import UTC, datetime
from unittest.mock import AsyncMock

import httpx
import pytest

from app.core.rate_limit import limiter
from app.core.settings import get_settings
from app.dependencies.services import get_chat_session_lifecycle_service
from app.exceptions.chat_session import (
    ChatSessionAccessDeniedError,
    ChatSessionAlreadyClosedError,
    ChatSessionNotFoundError,
)
from app.main import app
from app.services.chat_session_lifecycle import (
    BOUNDARY_RETAINED,
    ChatSessionClosure,
    ChatSessionCreation,
    ChatSessionLifecycleService,
    ChatSessionResolution,
)


@pytest.fixture(autouse=True)
def clear_overrides_and_limiter():
    limiter.reset()
    yield
    app.dependency_overrides.clear()
    limiter.reset()


async def _resolve_request(
    service: ChatSessionLifecycleService,
    *,
    headers: dict[str, str] | None = None,
    session_id: str = 'session-1',
    replacement_session_id: str = 'session-2',
) -> httpx.Response:
    app.dependency_overrides[
        get_chat_session_lifecycle_service
    ] = lambda: service
    request_headers = {
        'X-API-Key': get_settings().app.api_key.get_secret_value(),
    }
    request_headers.update(headers or {})
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport,
        base_url='http://test',
    ) as client:
        return await client.post(
            '/api/v1/chat/sessions/resolve',
            headers=request_headers,
            json={
                'session_id': session_id,
                'replacement_session_id': replacement_session_id,
            },
        )


@pytest.mark.asyncio
async def test_resolve_chat_session_endpoint_returns_contract_and_headers():
    service = AsyncMock(spec=ChatSessionLifecycleService)
    service.resolve_session.return_value = ChatSessionResolution(
        session_id='session-1',
        previous_session_id=None,
        boundary=BOUNDARY_RETAINED,
        session_ttl_seconds=86400,
    )

    response = await _resolve_request(
        service,
        headers={
            'X-Vera-User-ID': 'user-1',
            'X-Vera-Anonymous-Token-Hash': 'a' * 64,
            'X-Vera-Refreshed-Anonymous-Token-Hash': 'b' * 64,
            'X-Vera-Replacement-Anonymous-Token-Hash': 'c' * 64,
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        'session_id': 'session-1',
        'previous_session_id': None,
        'boundary': 'retained',
        'session_ttl_seconds': 86400,
    }
    service.resolve_session.assert_awaited_once_with(
        session_id='session-1',
        replacement_session_id='session-2',
        user_id='user-1',
        anonymous_token_hash='a' * 64,
        refreshed_anonymous_token_hash='b' * 64,
        replacement_anonymous_token_hash='c' * 64,
    )


@pytest.mark.asyncio
async def test_resolve_chat_session_endpoint_maps_owner_error_to_403():
    service = AsyncMock(spec=ChatSessionLifecycleService)
    service.resolve_session.side_effect = ChatSessionAccessDeniedError

    response = await _resolve_request(
        service,
        headers={'X-Vera-Anonymous-Token-Hash': 'a' * 64},
    )

    assert response.status_code == 403
    assert response.json() == {'detail': 'Нет доступа к этой сессии.'}


@pytest.mark.asyncio
async def test_resolve_chat_session_endpoint_rejects_same_replacement_id():
    service = AsyncMock(spec=ChatSessionLifecycleService)

    response = await _resolve_request(
        service,
        headers={'X-Vera-User-ID': 'user-1'},
        replacement_session_id='session-1',
    )

    assert response.status_code == 422
    service.resolve_session.assert_not_awaited()


@pytest.mark.asyncio
async def test_create_chat_session_endpoint_returns_contract_and_headers():
    service = AsyncMock(spec=ChatSessionLifecycleService)
    service.create_session.return_value = ChatSessionCreation(
        session_id='session-1',
        session_ttl_seconds=86400,
    )
    app.dependency_overrides[
        get_chat_session_lifecycle_service
    ] = lambda: service
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(
        transport=transport,
        base_url='http://test',
    ) as client:
        response = await client.post(
            '/api/v1/chat/sessions',
            headers={
                'X-API-Key': get_settings().app.api_key.get_secret_value(),
                'X-Vera-User-ID': 'user-1',
                'X-Vera-Anonymous-Token-Hash': 'a' * 64,
            },
            json={'session_id': 'session-1'},
        )

    assert response.status_code == 200
    assert response.json() == {
        'session_id': 'session-1',
        'session_ttl_seconds': 86400,
    }
    service.create_session.assert_awaited_once_with(
        session_id='session-1',
        user_id='user-1',
        anonymous_token_hash='a' * 64,
    )


@pytest.mark.asyncio
async def test_create_chat_session_endpoint_maps_closed_conflict_to_409():
    service = AsyncMock(spec=ChatSessionLifecycleService)
    service.create_session.side_effect = ChatSessionAlreadyClosedError
    app.dependency_overrides[
        get_chat_session_lifecycle_service
    ] = lambda: service
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(
        transport=transport,
        base_url='http://test',
    ) as client:
        response = await client.post(
            '/api/v1/chat/sessions',
            headers={
                'X-API-Key': get_settings().app.api_key.get_secret_value(),
                'X-Vera-User-ID': 'user-1',
            },
            json={'session_id': 'session-1'},
        )

    assert response.status_code == 409
    assert response.json() == {
        'detail': 'Сессия уже закрыта. Используйте новый идентификатор.'
    }


@pytest.mark.asyncio
async def test_close_chat_session_endpoint_returns_persisted_closed_at():
    closed_at = datetime.now(UTC)
    service = AsyncMock(spec=ChatSessionLifecycleService)
    service.close_session.return_value = ChatSessionClosure(
        session_id='session-1',
        closed_at=closed_at,
    )
    app.dependency_overrides[
        get_chat_session_lifecycle_service
    ] = lambda: service
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(
        transport=transport,
        base_url='http://test',
    ) as client:
        response = await client.post(
            '/api/v1/chat/sessions/session-1/close',
            headers={
                'X-API-Key': get_settings().app.api_key.get_secret_value(),
                'X-Vera-Anonymous-Token-Hash': 'a' * 64,
            },
        )

    assert response.status_code == 200
    assert response.json() == {
        'session_id': 'session-1',
        'closed_at': closed_at.isoformat().replace('+00:00', 'Z'),
    }
    service.close_session.assert_awaited_once_with(
        session_id='session-1',
        user_id=None,
        anonymous_token_hash='a' * 64,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('service_error', 'expected_status'),
    [
        (ChatSessionNotFoundError('missing-session'), 404),
        (ChatSessionAccessDeniedError(), 403),
    ],
)
async def test_close_chat_session_endpoint_maps_owner_and_missing_errors(
    service_error: Exception,
    expected_status: int,
):
    service = AsyncMock(spec=ChatSessionLifecycleService)
    service.close_session.side_effect = service_error
    app.dependency_overrides[
        get_chat_session_lifecycle_service
    ] = lambda: service
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(
        transport=transport,
        base_url='http://test',
    ) as client:
        response = await client.post(
            '/api/v1/chat/sessions/session-1/close',
            headers={
                'X-API-Key': get_settings().app.api_key.get_secret_value(),
                'X-Vera-User-ID': 'user-1',
            },
        )

    assert response.status_code == expected_status
