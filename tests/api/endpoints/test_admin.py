from uuid import uuid4

import httpx
import pytest

from app.admin.views import _fmt_feedback_turn
from app.db.models.chat_session import ChatSession
from app.db.models.chat_turn import ChatTurn
from app.db.models.message_feedback import MessageFeedback
from app.main import app


@pytest.mark.parametrize(
    'path',
    [
        '/admin/chat-session/list',
        '/admin/chat-turn/list',
        '/admin/message-feedback/list',
        '/admin/session-feedback/list',
        '/admin/session-detail',
    ],
)
@pytest.mark.asyncio
async def test_admin_views_redirect_unauthenticated_request(path: str):
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(
        transport=transport,
        base_url='http://test',
        follow_redirects=False,
    ) as client:
        response = await client.get(path)

    assert response.status_code in (302, 303)
    assert '/admin/login' in response.headers['location']


def test_message_feedback_links_to_its_turn_in_full_session():
    chat_session = ChatSession(id=uuid4(), session_id='session-1')
    chat_turn = ChatTurn(
        id=uuid4(),
        chat_session=chat_session,
        chat_session_id=chat_session.id,
        request_id='request-1',
        sequence_number=1,
        question='Вопрос',
        status='completed',
    )
    feedback = MessageFeedback(
        chat_turn=chat_turn,
        chat_turn_id=chat_turn.id,
        value='down',
    )

    link = str(_fmt_feedback_turn(feedback, 'chat_turn'))

    assert '/admin/session-detail?session_id=session-1&request_id=request-1' in link
