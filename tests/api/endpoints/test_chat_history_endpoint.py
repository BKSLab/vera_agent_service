from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

from app.core.rate_limit import limiter
from app.core.settings import get_settings
from app.dependencies.services import (
    get_chat_history_service,
    get_chat_session_lifecycle_service,
)
from app.exceptions.chat_session import (
    ChatSessionAccessDeniedError,
    ChatSessionNotFoundError,
)
from app.main import app
from app.services.chat_history import ChatHistoryPage, ChatHistoryService
from app.services.chat_session_lifecycle import ChatSessionLifecycleService


@pytest.fixture(autouse=True)
def clear_overrides_and_limiter():
    limiter.reset()
    yield
    app.dependency_overrides.clear()
    limiter.reset()


@pytest.mark.asyncio
async def test_current_chat_session_endpoint_returns_user_session():
    service = AsyncMock(spec=ChatSessionLifecycleService)
    service.get_current_user_session.return_value = SimpleNamespace(
        session_id='session-1'
    )
    app.dependency_overrides[
        get_chat_session_lifecycle_service
    ] = lambda: service
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(
        transport=transport,
        base_url='http://test',
    ) as client:
        response = await client.get(
            '/api/v1/chat/sessions/current',
            headers={
                'X-API-Key': get_settings().app.api_key.get_secret_value(),
                'X-Vera-User-ID': 'user-1',
            },
        )

    assert response.status_code == 200
    assert response.json() == {'session_id': 'session-1'}
    service.get_current_user_session.assert_awaited_once_with('user-1')


@pytest.mark.asyncio
async def test_current_chat_session_endpoint_returns_null_for_new_user():
    service = AsyncMock(spec=ChatSessionLifecycleService)
    service.get_current_user_session.return_value = None
    app.dependency_overrides[
        get_chat_session_lifecycle_service
    ] = lambda: service
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(
        transport=transport,
        base_url='http://test',
    ) as client:
        response = await client.get(
            '/api/v1/chat/sessions/current',
            headers={
                'X-API-Key': get_settings().app.api_key.get_secret_value(),
                'X-Vera-User-ID': 'user-1',
            },
        )

    assert response.status_code == 200
    assert response.json() == {'session_id': None}


@pytest.mark.parametrize(
    ('user_id_length', 'expected_status'),
    [(255, 200), (256, 422)],
)
@pytest.mark.asyncio
async def test_current_chat_session_endpoint_validates_user_id_length(
    user_id_length: int,
    expected_status: int,
):
    service = AsyncMock(spec=ChatSessionLifecycleService)
    service.get_current_user_session.return_value = None
    app.dependency_overrides[
        get_chat_session_lifecycle_service
    ] = lambda: service
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(
        transport=transport,
        base_url='http://test',
    ) as client:
        response = await client.get(
            '/api/v1/chat/sessions/current',
            headers={
                'X-API-Key': get_settings().app.api_key.get_secret_value(),
                'X-Vera-User-ID': 'u' * user_id_length,
            },
        )

    assert response.status_code == expected_status


@pytest.mark.asyncio
async def test_chat_history_endpoint_returns_turn_contract():
    now = datetime.now(UTC)
    service = AsyncMock(spec=ChatHistoryService)
    service.get_history.return_value = ChatHistoryPage(
        turns=[
            SimpleNamespace(
                request_id='request-1',
                sequence_number=1,
                question='Вопрос',
                answer='Ответ',
                status='completed',
                feedback=SimpleNamespace(value='up'),
                sources=[{'chunk_id': 'c1'}],
                created_at=now,
                completed_at=now,
            )
        ],
        next_before_sequence=1,
    )
    app.dependency_overrides[get_chat_history_service] = lambda: service
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(
        transport=transport,
        base_url='http://test',
    ) as client:
        response = await client.get(
            '/api/v1/chat/sessions/session-1/history',
            headers={
                'X-API-Key': get_settings().app.api_key.get_secret_value(),
            },
        )

    assert response.status_code == 200
    assert response.json() == {
        'session_id': 'session-1',
        'turns': [
            {
                'request_id': 'request-1',
                'sequence_number': 1,
                'question': 'Вопрос',
                'answer': 'Ответ',
                'status': 'completed',
                'feedback_value': 'up',
                'used_knowledge_base': True,
                'created_at': now.isoformat().replace('+00:00', 'Z'),
                'completed_at': now.isoformat().replace('+00:00', 'Z'),
            }
        ],
        'next_before_sequence': 1,
    }
    service.get_history.assert_awaited_once_with(
        'session-1',
        user_id=None,
        anonymous_token_hash=None,
        limit=30,
        before_sequence=None,
    )


@pytest.mark.asyncio
async def test_chat_history_endpoint_reports_no_knowledge_base_use_without_sources():
    """Пустые источники — прямой ответ либо честный отказ поиска. Кнопка
    «Объяснить проще» после перезагрузки страницы не должна появляться там,
    где её не было в живом диалоге."""
    now = datetime.now(UTC)
    service = AsyncMock(spec=ChatHistoryService)
    service.get_history.return_value = ChatHistoryPage(
        turns=[
            SimpleNamespace(
                request_id='request-1',
                sequence_number=1,
                question='Привет',
                answer='Здравствуйте!',
                status='completed',
                feedback=None,
                sources=[],
                created_at=now,
                completed_at=now,
            )
        ],
        next_before_sequence=None,
    )
    app.dependency_overrides[get_chat_history_service] = lambda: service
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(
        transport=transport,
        base_url='http://test',
    ) as client:
        response = await client.get(
            '/api/v1/chat/sessions/session-1/history',
            headers={
                'X-API-Key': get_settings().app.api_key.get_secret_value(),
            },
        )

    assert response.status_code == 200
    assert response.json()['turns'][0]['used_knowledge_base'] is False


@pytest.mark.asyncio
async def test_chat_history_endpoint_returns_404_for_unknown_session():
    service = AsyncMock(spec=ChatHistoryService)
    service.get_history.side_effect = ChatSessionNotFoundError('missing-session')
    app.dependency_overrides[get_chat_history_service] = lambda: service
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(
        transport=transport,
        base_url='http://test',
    ) as client:
        response = await client.get(
            '/api/v1/chat/sessions/missing-session/history',
            headers={
                'X-API-Key': get_settings().app.api_key.get_secret_value(),
            },
        )

    assert response.status_code == 404
    assert response.json() == {'detail': 'Сессия missing-session не найдена.'}


@pytest.mark.parametrize(
    ('user_id_length', 'expected_status'),
    [(255, 200), (256, 422)],
)
@pytest.mark.asyncio
async def test_chat_history_endpoint_validates_user_id_length(
    user_id_length: int,
    expected_status: int,
):
    service = AsyncMock(spec=ChatHistoryService)
    service.get_history.return_value = ChatHistoryPage(
        turns=[],
        next_before_sequence=None,
    )
    app.dependency_overrides[get_chat_history_service] = lambda: service
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(
        transport=transport,
        base_url='http://test',
    ) as client:
        response = await client.get(
            '/api/v1/chat/sessions/session-1/history',
            headers={
                'X-API-Key': get_settings().app.api_key.get_secret_value(),
                'X-Vera-User-ID': 'u' * user_id_length,
            },
        )

    assert response.status_code == expected_status


@pytest.mark.asyncio
async def test_chat_history_endpoint_rejects_another_owner():
    service = AsyncMock(spec=ChatHistoryService)
    service.get_history.side_effect = ChatSessionAccessDeniedError
    app.dependency_overrides[get_chat_history_service] = lambda: service
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(
        transport=transport,
        base_url='http://test',
    ) as client:
        response = await client.get(
            '/api/v1/chat/sessions/session-1/history',
            headers={
                'X-API-Key': get_settings().app.api_key.get_secret_value(),
                'X-Vera-User-ID': 'another-user',
            },
        )

    assert response.status_code == 403


@pytest.mark.asyncio
async def test_chat_history_endpoint_rejects_missing_api_key():
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(
        transport=transport,
        base_url='http://test',
    ) as client:
        response = await client.get(
            '/api/v1/chat/sessions/session-1/history',
        )

    assert response.status_code == 422
