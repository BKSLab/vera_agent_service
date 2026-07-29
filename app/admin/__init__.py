from pathlib import Path

from fastapi import FastAPI
from sqladmin import Admin
from sqlalchemy.ext.asyncio import AsyncEngine

from app.admin.auth import AdminLoginAuth
from app.admin.views import (
    ChatSessionAdmin,
    ChatTurnAdmin,
    DialogueSearchView,
    MessageFeedbackAdmin,
    SessionDetailView,
    SessionFeedbackAdmin,
)
from app.core.settings import get_settings

_TEMPLATES_DIR = str(Path(__file__).parent.parent / 'templates')


def create_admin(app: FastAPI, engine: AsyncEngine) -> Admin:
    """Создаёт и регистрирует SQLAdmin для Agent Service."""
    settings = get_settings()
    admin = Admin(
        app=app,
        engine=engine,
        authentication_backend=AdminLoginAuth(
            secret_key=settings.app.secret_key.get_secret_value(),
            https_only=settings.app.admin_session_https_only,
        ),
        title='Vera Agent Service — Admin',
        base_url='/admin',
        templates_dir=_TEMPLATES_DIR,
    )
    admin.add_view(ChatSessionAdmin)
    admin.add_view(ChatTurnAdmin)
    admin.add_view(MessageFeedbackAdmin)
    admin.add_view(SessionFeedbackAdmin)
    admin.add_view(DialogueSearchView)
    admin.add_view(SessionDetailView)
    return admin
