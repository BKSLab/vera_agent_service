import json
from typing import Any
from urllib.parse import urlencode

from markupsafe import Markup, escape
from sqladmin import BaseView, ModelView, expose
from sqlalchemy import select
from sqlalchemy.orm import selectinload
from starlette.requests import Request

from app.core.settings import get_settings
from app.db.models.chat_session import ChatSession
from app.db.models.chat_turn import ChatTurn
from app.db.models.message_feedback import MessageFeedback
from app.db.models.session_feedback import SessionFeedback
from app.db.session import async_session_factory

REVIEW_STATUS_CHOICES = [
    ('new', 'Новый'),
    ('in_review', 'На проверке'),
    ('resolved', 'Решён'),
    ('dismissed', 'Отклонён'),
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
    column_searchable_list = [ChatTurn.request_id, ChatTurn.question, ChatTurn.answer]
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
    column_searchable_list = [SessionFeedback.submission_id, SessionFeedback.comment]
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
