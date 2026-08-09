import httpx
import pytest

from app.core.rate_limit import limiter
from app.core.settings import get_settings
from app.db.models.chat_session import ChatSession
from app.db.models.chat_turn import STATUS_COMPLETED, ChatTurn
from app.dependencies.services import get_chat_history_service
from app.main import app
from app.repositories.chat_session import ChatSessionRepository
from app.repositories.chat_turn import ChatTurnRepository
from app.services.chat_history import ChatHistoryService

pytestmark = pytest.mark.integration


@pytest.fixture(autouse=True)
def clear_overrides_and_limiter():
    limiter.reset()
    yield
    app.dependency_overrides.clear()
    limiter.reset()


async def _request_history(
    service: ChatHistoryService,
    session_id: str,
    user_id: str,
) -> httpx.Response:
    app.dependency_overrides[get_chat_history_service] = lambda: service
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport,
        base_url='http://test',
    ) as client:
        return await client.get(
            f'/api/v1/chat/sessions/{session_id}/history',
            headers={
                'X-API-Key': get_settings().app.api_key.get_secret_value(),
                'X-Vera-User-ID': user_id,
            },
        )


async def test_missing_session_returns_404_on_real_postgresql(db_session):
    service = ChatHistoryService(
        ChatSessionRepository(db_session),
        ChatTurnRepository(db_session),
    )

    response = await _request_history(service, 'missing-session', 'user-1')

    assert response.status_code == 404
    assert response.json() == {'detail': 'Сессия missing-session не найдена.'}


async def test_foreign_session_returns_403_on_real_postgresql(db_session):
    session_repository = ChatSessionRepository(db_session)
    await session_repository.save(
        ChatSession(session_id='foreign-session', user_id='owner-1')
    )
    service = ChatHistoryService(
        session_repository,
        ChatTurnRepository(db_session),
    )

    response = await _request_history(service, 'foreign-session', 'owner-2')

    assert response.status_code == 403
    assert response.json() == {'detail': 'Нет доступа к этой сессии.'}


async def test_owned_session_returns_200_on_real_postgresql(db_session):
    session_repository = ChatSessionRepository(db_session)
    turn_repository = ChatTurnRepository(db_session)
    chat_session = await session_repository.save(
        ChatSession(session_id='owned-session', user_id='owner-1')
    )
    await turn_repository.save(
        ChatTurn(
            request_id='owned-request',
            chat_session_id=chat_session.id,
            sequence_number=1,
            user_id='owner-1',
            question='Вопрос',
            answer='Ответ',
            status=STATUS_COMPLETED,
        )
    )
    service = ChatHistoryService(session_repository, turn_repository)

    response = await _request_history(service, 'owned-session', 'owner-1')

    assert response.status_code == 200
    payload = response.json()
    assert payload['session_id'] == 'owned-session'
    assert payload['next_before_sequence'] is None
    assert [turn['request_id'] for turn in payload['turns']] == ['owned-request']
