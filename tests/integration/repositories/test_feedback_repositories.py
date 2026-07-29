import pytest

from app.db.models.chat_session import ChatSession
from app.db.models.chat_turn import ChatTurn
from app.repositories.chat_session import ChatSessionRepository
from app.repositories.chat_turn import ChatTurnRepository
from app.repositories.message_feedback import MessageFeedbackRepository
from app.repositories.session_feedback import SessionFeedbackRepository
from app.services.message_feedback import MessageFeedbackService
from app.services.session_feedback import SessionFeedbackService

pytestmark = pytest.mark.integration


async def test_repositories_persist_turn_and_feedback_without_duplicates(db_session):
    session_repository = ChatSessionRepository(db_session)
    turn_repository = ChatTurnRepository(db_session)
    message_feedback_repository = MessageFeedbackRepository(db_session)
    session_feedback_repository = SessionFeedbackRepository(db_session)

    chat_session = await session_repository.save(ChatSession(session_id='session-1'))
    chat_turn = await turn_repository.save(
        ChatTurn(
            request_id='request-1',
            chat_session_id=chat_session.id,
            sequence_number=1,
            question='Вопрос',
            answer='Ответ',
            status='completed',
        )
    )

    message_service = MessageFeedbackService(
        turn_repository,
        message_feedback_repository,
    )
    first_rating = await message_service.upsert_feedback(
        session_id='session-1',
        request_id='request-1',
        value='down',
    )
    second_rating = await message_service.upsert_feedback(
        session_id='session-1',
        request_id='request-1',
        value='up',
    )
    await turn_repository.save(
        ChatTurn(
            request_id='request-2',
            chat_session_id=chat_session.id,
            sequence_number=2,
            question='Второй вопрос',
            answer='Второй ответ',
            status='completed',
        )
    )
    history = await turn_repository.list_by_chat_session_id(chat_session.id)

    session_service = SessionFeedbackService(
        session_repository,
        session_feedback_repository,
    )
    first_form = await session_service.create_feedback(
        session_id='session-1',
        submission_id='submission-1',
        audience='employer',
        usefulness=5,
        trust=4,
        comment='Комментарий',
        contact_email='user@example.ru',
    )
    second_form = await session_service.create_feedback(
        session_id='session-1',
        submission_id='submission-1',
        audience='employer',
        usefulness=5,
        trust=4,
        comment='Комментарий',
        contact_email='user@example.ru',
    )

    assert first_rating.id == second_rating.id
    assert second_rating.value == 'up'
    assert [turn.request_id for turn in history] == ['request-1', 'request-2']
    assert history[0].feedback.value == 'up'
    assert first_form.id == second_form.id
    assert chat_turn.request_id == 'request-1'


async def test_repositories_use_russian_full_text_search_and_filters(db_session):
    session_repository = ChatSessionRepository(db_session)
    turn_repository = ChatTurnRepository(db_session)
    message_feedback_repository = MessageFeedbackRepository(db_session)
    session_feedback_repository = SessionFeedbackRepository(db_session)

    chat_session = await session_repository.save(
        ChatSession(session_id='search-session')
    )
    matching_turn = await turn_repository.save(
        ChatTurn(
            request_id='search-request',
            chat_session_id=chat_session.id,
            sequence_number=1,
            question='Как оформить трудовой договор?',
            answer='Работодатель оформляет трудовой договор письменно.',
            status='completed',
        )
    )
    await turn_repository.save(
        ChatTurn(
            request_id='unrelated-request',
            chat_session_id=chat_session.id,
            sequence_number=2,
            question='Как восстановить пароль?',
            answer='Обратитесь в поддержку.',
            status='completed',
        )
    )
    await MessageFeedbackService(
        turn_repository,
        message_feedback_repository,
    ).upsert_feedback(
        session_id='search-session',
        request_id='search-request',
        value='down',
    )
    await SessionFeedbackService(
        session_repository,
        session_feedback_repository,
    ).create_feedback(
        session_id='search-session',
        submission_id='search-submission',
        audience='employer',
        usefulness=3,
        trust=3,
        comment='Не хватило информации о трудоустройстве.',
        contact_email=None,
    )

    turn_results = await turn_repository.search(
        search_query='трудового договора',
        turn_status='completed',
        rating='down',
        audience='employer',
        created_from=None,
        created_to=None,
        limit=50,
    )
    comment_results = await session_feedback_repository.search_comments(
        search_query='информация о трудоустройстве',
        turn_status='completed',
        rating='down',
        audience='employer',
        created_from=None,
        created_to=None,
        limit=50,
    )

    assert [result.chat_turn.id for result in turn_results] == [matching_turn.id]
    assert len(comment_results) == 1
    assert comment_results[0].feedback.submission_id == 'search-submission'
