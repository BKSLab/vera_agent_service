import json
from datetime import UTC, date, datetime, time, timedelta
from typing import Any
from urllib.parse import urlencode

from markupsafe import Markup, escape
from sqladmin import BaseView, ModelView, expose
from sqlalchemy import select
from sqlalchemy.orm import selectinload
from starlette.requests import Request

from app.admin.services import build_dialogue_search_service
from app.core.settings import get_settings
from app.db.models.chat_session import ChatSession
from app.db.models.chat_turn import ChatTurn
from app.db.models.message_feedback import MessageFeedback
from app.db.models.session_feedback import SessionFeedback
from app.db.session import async_session_factory
from app.exceptions.dialogue_search import DialogueSearchServiceError

REVIEW_STATUS_CHOICES = [
    ('new', 'Новый'),
    ('in_review', 'На проверке'),
    ('resolved', 'Решён'),
    ('dismissed', 'Отклонён'),
]
TURN_STATUS_CHOICES = [
    ('processing', 'В обработке'),
    ('completed', 'Завершён'),
    ('failed', 'Ошибка'),
    ('delivery_unconfirmed', 'Доставка не подтверждена'),
]
RATING_CHOICES = [
    ('up', 'Положительная'),
    ('down', 'Отрицательная'),
    ('none', 'Без оценки'),
]
AUDIENCE_CHOICES = [
    ('seeker', 'Соискатель'),
    ('employer', 'Работодатель'),
    ('other', 'Другая'),
]


def _short_text(value: str | None, limit: int = 100) -> Markup:
    if not value:
        return Markup('<em>—</em>')
    text = value if len(value) <= limit else f'{value[:limit]}…'
    return Markup(f'<span title="{escape(value)}">{escape(text)}</span>')


def _fmt_question(model: ChatTurn, attr: str) -> Markup:
    return _short_text(model.question)


def _fmt_answer(model: ChatTurn, attr: str) -> Markup:
    return _short_text(model.answer)


def _fmt_json(model: Any, attr: str) -> Markup:
    value = getattr(model, attr, None)
    pretty = json.dumps(value, ensure_ascii=False, indent=2)
    return Markup(f'<pre style="white-space:pre-wrap;word-break:break-word">{escape(pretty)}</pre>')


def _session_link(session_id: str) -> Markup:
    query = urlencode({'session_id': session_id})
    return Markup(f'<a href="/admin/session-detail?{query}">{escape(session_id)}</a>')


def _session_detail_url(session_id: str, request_id: str | None = None) -> str:
    params = {'session_id': session_id}
    if request_id:
        params['request_id'] = request_id
    return f'/admin/session-detail?{urlencode(params)}'


def _parse_filter_date(value: str, *, end_of_day: bool = False) -> datetime | None:
    if not value:
        return None
    try:
        parsed = date.fromisoformat(value)
    except ValueError as error:
        raise ValueError('Дата должна быть указана в формате ГГГГ-ММ-ДД.') from error
    result = datetime.combine(parsed, time.min, tzinfo=UTC)
    return result + timedelta(days=1) if end_of_day else result


def _fmt_session_id(model: ChatSession, attr: str) -> Markup:
    return _session_link(model.session_id)


def _fmt_turn_session(model: ChatTurn, attr: str) -> Markup:
    return _session_link(model.chat_session.session_id)


def _fmt_feedback_turn(model: MessageFeedback, attr: str) -> Markup:
    turn = model.chat_turn
    query = urlencode({'session_id': turn.chat_session.session_id, 'request_id': turn.request_id})
    return Markup(f'<a href="/admin/session-detail?{query}">{escape(turn.request_id)}</a>')


def _fmt_feedback_session(model: SessionFeedback, attr: str) -> Markup:
    return _session_link(model.chat_session.session_id)


def _mask_email(email: str | None) -> str:
    if not email or '@' not in email:
        return '—'
    local, domain = email.split('@', 1)
    visible = local[:2]
    return f'{visible}***@{domain}'


def _fmt_masked_email(model: SessionFeedback, attr: str) -> Markup:
    return Markup(escape(_mask_email(model.contact_email)))


class ChatSessionAdmin(ModelView, model=ChatSession):
    """Read-only список постоянных сессий."""

    name = 'Сессия'
    name_plural = 'Сессии'
    icon = 'fa-solid fa-comments'

    column_list = [
        ChatSession.session_id,
        ChatSession.user_id,
        ChatSession.created_at,
        ChatSession.last_activity_at,
        ChatSession.closed_at,
    ]
    column_details_list = column_list
    column_searchable_list = [ChatSession.session_id, ChatSession.user_id]
    column_sortable_list = [
        ChatSession.created_at,
        ChatSession.last_activity_at,
    ]
    column_default_sort = [(ChatSession.last_activity_at, True)]
    column_formatters = {ChatSession.session_id: _fmt_session_id}
    column_formatters_detail = {ChatSession.session_id: _fmt_session_id}

    can_create = False
    can_edit = False
    can_delete = False


class ChatTurnAdmin(ModelView, model=ChatTurn):
    """Read-only список вопросов и фактически отправленных ответов."""

    name = 'Реплика'
    name_plural = 'Реплики'
    icon = 'fa-solid fa-message'

    column_list = [
        ChatTurn.request_id,
        ChatTurn.chat_session,
        ChatTurn.sequence_number,
        ChatTurn.question,
        ChatTurn.answer,
        ChatTurn.status,
        ChatTurn.latency_ms,
        ChatTurn.created_at,
    ]
    column_details_list = [
        ChatTurn.request_id,
        ChatTurn.chat_session,
        ChatTurn.sequence_number,
        ChatTurn.user_id,
        ChatTurn.question,
        ChatTurn.answer,
        ChatTurn.sources,
        ChatTurn.technical_metadata,
        ChatTurn.status,
        ChatTurn.safe_error,
        ChatTurn.started_at,
        ChatTurn.completed_at,
        ChatTurn.latency_ms,
    ]
    column_searchable_list = [ChatTurn.request_id]
    column_sortable_list = [ChatTurn.created_at, ChatTurn.completed_at, ChatTurn.latency_ms]
    column_default_sort = [(ChatTurn.created_at, True)]
    column_formatters = {
        ChatTurn.chat_session: _fmt_turn_session,
        ChatTurn.question: _fmt_question,
        ChatTurn.answer: _fmt_answer,
    }
    column_formatters_detail = {
        ChatTurn.chat_session: _fmt_turn_session,
        ChatTurn.sources: _fmt_json,
        ChatTurn.technical_metadata: _fmt_json,
    }

    can_create = False
    can_edit = False
    can_delete = False


class MessageFeedbackAdmin(ModelView, model=MessageFeedback):
    """Оценки ответов с редактируемыми экспертными полями."""

    name = 'Оценка ответа'
    name_plural = 'Оценки ответов'
    icon = 'fa-solid fa-thumbs-up'

    column_list = [
        MessageFeedback.value,
        MessageFeedback.chat_turn,
        MessageFeedback.review_status,
        MessageFeedback.tags,
        MessageFeedback.created_at,
        MessageFeedback.updated_at,
    ]
    column_details_list = [
        MessageFeedback.id,
        MessageFeedback.chat_turn,
        MessageFeedback.value,
        MessageFeedback.review_status,
        MessageFeedback.expert_note,
        MessageFeedback.tags,
        MessageFeedback.updated_by_admin,
        MessageFeedback.created_at,
        MessageFeedback.updated_at,
    ]
    column_sortable_list = [
        MessageFeedback.created_at,
        MessageFeedback.updated_at,
        MessageFeedback.review_status,
    ]
    column_default_sort = [(MessageFeedback.created_at, True)]
    column_formatters = {
        MessageFeedback.chat_turn: _fmt_feedback_turn,
        MessageFeedback.tags: _fmt_json,
    }
    column_formatters_detail = {
        MessageFeedback.chat_turn: _fmt_feedback_turn,
        MessageFeedback.tags: _fmt_json,
    }
    form_columns = [
        MessageFeedback.review_status,
        MessageFeedback.expert_note,
        MessageFeedback.tags,
    ]
    form_choices = {'review_status': REVIEW_STATUS_CHOICES}

    can_create = False
    can_edit = True
    can_delete = False

    async def on_model_change(
        self,
        data: dict,
        model: MessageFeedback,
        is_created: bool,
        request: Request,
    ) -> None:
        """Фиксирует автора изменения экспертных полей."""
        model.updated_by_admin = get_settings().app.admin_login


class SessionFeedbackAdmin(ModelView, model=SessionFeedback):
    """Развёрнутые отзывы с редактируемыми экспертными полями."""

    name = 'Отзыв по сессии'
    name_plural = 'Отзывы по сессиям'
    icon = 'fa-solid fa-clipboard-check'

    column_list = [
        SessionFeedback.chat_session,
        SessionFeedback.audience,
        SessionFeedback.usefulness,
        SessionFeedback.trust,
        SessionFeedback.contact_email,
        SessionFeedback.review_status,
        SessionFeedback.created_at,
    ]
    column_details_list = [
        SessionFeedback.id,
        SessionFeedback.chat_session,
        SessionFeedback.submission_id,
        SessionFeedback.audience,
        SessionFeedback.usefulness,
        SessionFeedback.trust,
        SessionFeedback.comment,
        SessionFeedback.contact_email,
        SessionFeedback.review_status,
        SessionFeedback.expert_note,
        SessionFeedback.tags,
        SessionFeedback.updated_by_admin,
        SessionFeedback.created_at,
        SessionFeedback.updated_at,
    ]
    column_searchable_list = [SessionFeedback.submission_id]
    column_sortable_list = [
        SessionFeedback.created_at,
        SessionFeedback.review_status,
    ]
    column_default_sort = [(SessionFeedback.created_at, True)]
    column_formatters = {
        SessionFeedback.chat_session: _fmt_feedback_session,
        SessionFeedback.contact_email: _fmt_masked_email,
    }
    column_formatters_detail = {
        SessionFeedback.chat_session: _fmt_feedback_session,
        SessionFeedback.tags: _fmt_json,
    }
    form_columns = [
        SessionFeedback.review_status,
        SessionFeedback.expert_note,
        SessionFeedback.tags,
    ]
    form_choices = {'review_status': REVIEW_STATUS_CHOICES}

    can_create = False
    can_edit = True
    can_delete = False

    async def on_model_change(
        self,
        data: dict,
        model: SessionFeedback,
        is_created: bool,
        request: Request,
    ) -> None:
        """Фиксирует автора изменения экспертных полей."""
        model.updated_by_admin = get_settings().app.admin_login


class SessionDetailView(BaseView):
    """Хронология одной сессии с оценками и развёрнутыми отзывами."""

    name = 'Просмотр сессии'
    icon = 'fa-solid fa-list'

    @expose('/session-detail', methods=['GET'])
    async def session_detail(self, request: Request) -> Any:
        session_id = (request.query_params.get('session_id') or '').strip()
        request_id = (request.query_params.get('request_id') or '').strip()
        chat_session = None
        if session_id:
            async with async_session_factory() as db_session:
                result = await db_session.execute(
                    select(ChatSession)
                    .options(
                        selectinload(ChatSession.turns).selectinload(ChatTurn.feedback),
                        selectinload(ChatSession.feedback_entries),
                    )
                    .where(ChatSession.session_id == session_id)
                )
                chat_session = result.unique().scalar_one_or_none()

        return await self.templates.TemplateResponse(
            request,
            'session_detail.html',
            {
                'chat_session': chat_session,
                'session_id': session_id,
                'request_id': request_id,
            },
        )


class DialogueSearchView(BaseView):
    """Полнотекстовый поиск по постоянному архиву диалогов и отзывов."""

    name = 'Поиск по диалогам'
    icon = 'fa-solid fa-magnifying-glass'

    @expose('/dialogue-search', methods=['GET'])
    async def dialogue_search(self, request: Request) -> Any:
        search_query = (request.query_params.get('query') or '').strip()
        turn_status = (request.query_params.get('status') or '').strip()
        rating = (request.query_params.get('rating') or '').strip()
        audience = (request.query_params.get('audience') or '').strip()
        date_from = (request.query_params.get('date_from') or '').strip()
        date_to = (request.query_params.get('date_to') or '').strip()

        context: dict[str, Any] = {
            'query': search_query,
            'selected_status': turn_status,
            'selected_rating': rating,
            'selected_audience': audience,
            'date_from': date_from,
            'date_to': date_to,
            'turn_status_choices': TURN_STATUS_CHOICES,
            'rating_choices': RATING_CHOICES,
            'audience_choices': AUDIENCE_CHOICES,
            'session_detail_url': _session_detail_url,
            'searched': False,
        }
        has_criteria = any(
            (search_query, turn_status, rating, audience, date_from, date_to)
        )
        if not has_criteria:
            return await self.templates.TemplateResponse(
                request,
                'dialogue_search.html',
                context,
            )

        try:
            allowed_statuses = {value for value, _ in TURN_STATUS_CHOICES}
            allowed_ratings = {value for value, _ in RATING_CHOICES}
            allowed_audiences = {value for value, _ in AUDIENCE_CHOICES}
            if turn_status and turn_status not in allowed_statuses:
                raise ValueError('Недопустимый статус реплики.')
            if rating and rating not in allowed_ratings:
                raise ValueError('Недопустимое значение оценки.')
            if audience and audience not in allowed_audiences:
                raise ValueError('Недопустимая аудитория.')

            created_from = _parse_filter_date(date_from)
            created_to = _parse_filter_date(date_to, end_of_day=True)
            if created_from and created_to and created_from >= created_to:
                raise ValueError('Дата начала периода должна быть не позже даты окончания.')

            async with build_dialogue_search_service() as service:
                context['results'] = await service.search(
                    search_query=search_query or None,
                    turn_status=turn_status or None,
                    rating=rating or None,
                    audience=audience or None,
                    created_from=created_from,
                    created_to=created_to,
                )
            context['searched'] = True
        except ValueError as error:
            context['error'] = str(error)
        except DialogueSearchServiceError as error:
            context['error'] = error.detail

        return await self.templates.TemplateResponse(
            request,
            'dialogue_search.html',
            context,
        )
