from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

from sqlalchemy import DateTime, String, UniqueConstraint, Uuid, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.models.base import Base

if TYPE_CHECKING:
    from app.db.models.chat_turn import ChatTurn
    from app.db.models.session_feedback import SessionFeedback


class ChatSession(Base):
    """Постоянная запись одной сессии диалога с Верой."""

    __tablename__ = 'vera_chat_sessions'
    __table_args__ = (
        UniqueConstraint('session_id', name='uq_vera_chat_sessions_session_id'),
    )

    id: Mapped[UUID] = mapped_column(
        Uuid,
        primary_key=True,
        default=uuid4,
        doc='Внутренний идентификатор сессии.',
        comment='Внутренний UUID сессии диалога.',
    )
    session_id: Mapped[str] = mapped_column(
        String(length=100),
        nullable=False,
        index=True,
        doc='Внешний идентификатор диалога.',
        comment='Идентификатор сессии, поступивший от сайта.',
    )
    user_id: Mapped[str | None] = mapped_column(
        String(length=255),
        nullable=True,
        index=True,
        doc='Идентификатор авторизованного пользователя.',
        comment='Идентификатор пользователя сайта, если он был авторизован.',
    )
    anonymous_token_hash: Mapped[str | None] = mapped_column(
        String(length=64),
        nullable=True,
        index=True,
        doc='SHA-256 подписанного токена анонимной сессии.',
        comment='Хеш серверного токена владельца анонимной сессии.',
    )
    service_metadata: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        doc='Безопасные служебные метаданные сессии.',
        comment='Служебные метаданные без текстов диалога и персональных данных.',
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        doc='Дата создания сессии.',
        comment='Момент создания постоянной записи сессии.',
    )
    last_activity_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
        index=True,
        doc='Дата последней активности.',
        comment='Момент последнего сохранённого запроса в сессии.',
    )
    closed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        doc='Дата закрытия сессии.',
        comment='Момент явного закрытия сессии.',
    )

    turns: Mapped[list[ChatTurn]] = relationship(
        back_populates='chat_session',
        cascade='all, delete-orphan',
        order_by='ChatTurn.sequence_number',
        lazy='raise',
    )
    feedback_entries: Mapped[list[SessionFeedback]] = relationship(
        back_populates='chat_session',
        cascade='all, delete-orphan',
        order_by='SessionFeedback.created_at',
        lazy='raise',
    )

    def __repr__(self) -> str:
        return f"<ChatSession(id={self.id}, session_id='{self.session_id}')>"
