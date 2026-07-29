from unittest.mock import AsyncMock

import pytest

from app.exceptions.chat_turn import ChatTurnRepositoryError
from app.exceptions.dialogue_search import DialogueSearchServiceError
from app.repositories.chat_turn import ChatTurnRepository
from app.repositories.session_feedback import SessionFeedbackRepository
from app.services.dialogue_search import DialogueSearchService


@pytest.mark.asyncio
async def test_dialogue_search_combines_turns_and_comments():
    turn_repository = AsyncMock(spec=ChatTurnRepository)
    session_feedback_repository = AsyncMock(spec=SessionFeedbackRepository)
    turn_repository.search.return_value = ['turn-result']
    session_feedback_repository.search_comments.return_value = ['comment-result']
    service = DialogueSearchService(
        chat_turn_repository=turn_repository,
        session_feedback_repository=session_feedback_repository,
    )

    result = await service.search(
        search_query='трудовой договор',
        turn_status='completed',
        rating='down',
        audience='employer',
        created_from=None,
        created_to=None,
    )

    assert result.turns == ['turn-result']
    assert result.comments == ['comment-result']
    turn_repository.search.assert_awaited_once_with(
        search_query='трудовой договор',
        turn_status='completed',
        rating='down',
        audience='employer',
        created_from=None,
        created_to=None,
        limit=50,
    )
    session_feedback_repository.search_comments.assert_awaited_once_with(
        search_query='трудовой договор',
        turn_status='completed',
        rating='down',
        audience='employer',
        created_from=None,
        created_to=None,
        limit=50,
    )


@pytest.mark.asyncio
async def test_dialogue_search_wraps_repository_error():
    turn_repository = AsyncMock(spec=ChatTurnRepository)
    session_feedback_repository = AsyncMock(spec=SessionFeedbackRepository)
    turn_repository.search.side_effect = ChatTurnRepositoryError
    service = DialogueSearchService(
        chat_turn_repository=turn_repository,
        session_feedback_repository=session_feedback_repository,
    )

    with pytest.raises(DialogueSearchServiceError) as exc_info:
        await service.search(
            search_query='ошибка',
            turn_status=None,
            rating=None,
            audience=None,
            created_from=None,
            created_to=None,
        )

    assert exc_info.value.status_code == 500
