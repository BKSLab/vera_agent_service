from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import httpx
import pytest

from app.core.rate_limit import limiter
from app.core.settings import get_settings
from app.dependencies.services import (
    get_message_feedback_service,
    get_session_feedback_service,
)
from app.main import app
from app.services.message_feedback import MessageFeedbackService
from app.services.session_feedback import SessionFeedbackService


@pytest.fixture(autouse=True)
def clear_overrides_and_limiter():
    limiter.reset()
    yield
    app.dependency_overrides.clear()
    limiter.reset()


@pytest.mark.asyncio
async def test_message_feedback_endpoint_upserts_feedback():
    now = datetime.now(UTC)
    service = AsyncMock(spec=MessageFeedbackService)
    service.upsert_feedback.return_value = SimpleNamespace(
        id=uuid4(),
        value='down',
        review_status='new',
        created_at=now,
        updated_at=now,
    )
    app.dependency_overrides[get_message_feedback_service] = lambda: service
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url='http://test') as client:
        response = await client.put(
            '/api/v1/feedback/message',
            headers={'X-API-Key': get_settings().app.api_key.get_secret_value()},
            json={
                'session_id': 'session-1',
                'request_id': 'request-1',
                'value': 'down',
            },
        )

    assert response.status_code == 200
    assert response.json()['value'] == 'down'
    service.upsert_feedback.assert_awaited_once()


@pytest.mark.parametrize(
    ('user_id_length', 'expected_status'),
    [(255, 200), (256, 422)],
)
@pytest.mark.asyncio
async def test_message_feedback_endpoint_validates_user_id_length(
    user_id_length: int,
    expected_status: int,
):
    now = datetime.now(UTC)
    service = AsyncMock(spec=MessageFeedbackService)
    service.upsert_feedback.return_value = SimpleNamespace(
        id=uuid4(),
        value='down',
        review_status='new',
        created_at=now,
        updated_at=now,
    )
    app.dependency_overrides[get_message_feedback_service] = lambda: service
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url='http://test') as client:
        response = await client.put(
            '/api/v1/feedback/message',
            headers={
                'X-API-Key': get_settings().app.api_key.get_secret_value(),
                'X-Vera-User-ID': 'u' * user_id_length,
            },
            json={
                'session_id': 'session-1',
                'request_id': 'request-1',
                'value': 'down',
            },
        )

    assert response.status_code == expected_status


@pytest.mark.asyncio
async def test_session_feedback_endpoint_creates_feedback():
    now = datetime.now(UTC)
    service = AsyncMock(spec=SessionFeedbackService)
    service.create_feedback.return_value = SimpleNamespace(
        id=uuid4(),
        submission_id='submission-1',
        review_status='new',
        created_at=now,
    )
    app.dependency_overrides[get_session_feedback_service] = lambda: service
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url='http://test') as client:
        response = await client.post(
            '/api/v1/feedback/session',
            headers={'X-API-Key': get_settings().app.api_key.get_secret_value()},
            json={
                'session_id': 'session-1',
                'submission_id': 'submission-1',
                'usefulness': 5,
                'trust': 4,
            },
        )

    assert response.status_code == 201
    assert response.json()['submission_id'] == 'submission-1'


@pytest.mark.parametrize(
    ('user_id_length', 'expected_status'),
    [(255, 201), (256, 422)],
)
@pytest.mark.asyncio
async def test_session_feedback_endpoint_validates_user_id_length(
    user_id_length: int,
    expected_status: int,
):
    now = datetime.now(UTC)
    service = AsyncMock(spec=SessionFeedbackService)
    service.create_feedback.return_value = SimpleNamespace(
        id=uuid4(),
        submission_id='submission-1',
        review_status='new',
        created_at=now,
    )
    app.dependency_overrides[get_session_feedback_service] = lambda: service
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url='http://test') as client:
        response = await client.post(
            '/api/v1/feedback/session',
            headers={
                'X-API-Key': get_settings().app.api_key.get_secret_value(),
                'X-Vera-User-ID': 'u' * user_id_length,
            },
            json={
                'session_id': 'session-1',
                'submission_id': 'submission-1',
                'usefulness': 5,
                'trust': 4,
            },
        )

    assert response.status_code == expected_status


@pytest.mark.asyncio
async def test_feedback_endpoint_rejects_missing_api_key():
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url='http://test') as client:
        response = await client.put(
            '/api/v1/feedback/message',
            json={
                'session_id': 'session-1',
                'request_id': 'request-1',
                'value': 'up',
            },
        )

    assert response.status_code == 422
