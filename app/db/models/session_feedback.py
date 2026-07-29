from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    Uuid,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.models.base import Base

if TYPE_CHECKING:
    from app.db.models.chat_session import ChatSession


class SessionFeedback(Base):
    """Развёрнутый пользовательский отзыв по сессии диалога."""

    __tablename__ = 'vera_session_feedback'
    __table_args__ = (
        CheckConstraint(
            'usefulness IS NULL OR usefulness BETWEEN 1 AND 5',
            name='ck_vera_session_feedback_usefulness',
        ),
        CheckConstraint(
            'trust IS NULL OR trust BETWEEN 1 AND 5',
            name='ck_vera_session_feedback_trust',
        ),
        UniqueConstraint(
            'submission_id',
            name='uq_vera_session_feedback_submission_id',
        ),
    )

    id: Mapped[UUID] = mapped_column(
        Uuid,
        primary_key=True,
        default=uuid4,
        doc='Внутренний идентификатор отзыва.',
        comment='Внутренний UUID развёрнутого отзыва.',
    )
    chat_session_id: Mapped[UUID] = mapped_column(
        ForeignKey('vera_chat_sessions.id', ondelete='CASCADE'),
        nullable=False,
        index=True,
        doc='Ссылка на сессию.',
        comment='Внешний ключ на сессию отзыва.',
    )
    submission_id: Mapped[str] = mapped_column(
        String(length=100),
        nullable=False,
        index=True,
        doc='Ключ идемпотентности.',
        comment='Идентификатор одной отправки анкеты.',
    )
    audience: Mapped[str | None] = mapped_column(
        String(length=50),
        nullable=True,
        doc='Аудитория пользователя.',
        comment='Категория пользователя из анкеты.',
    )
    usefulness: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
        doc='Оценка полезности.',
        comment='Оценка полезности от 1 до 5.',
    )
    trust: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
        doc='Оценка доверия.',
        comment='Оценка доверия от 1 до 5.',
    )
    comment: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        doc='Комментарий пользователя.',
        comment='Свободный комментарий пользователя.',
    )
    contact_email: Mapped[str | None] = mapped_column(
        String(length=320),
        nullable=True,
        doc='Контактный email.',
        comment='Добровольно указанный контактный email.',
    )
    review_status: Mapped[str] = mapped_column(
        String(length=20),
        nullable=False,
        default='new',
        index=True,
        doc='Статус экспертной проверки.',
        comment='Статус экспертной проверки.',
    )
    expert_note: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        doc='Внутренняя заметка эксперта.',
        comment='Служебная заметка, не доступная пользователю.',
    )
    tags: Mapped[list] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
        doc='Теги экспертной обработки.',
        comment='Список причин или категорий экспертной проверки.',
    )
    updated_by_admin: Mapped[str | None] = mapped_column(
        String(length=100),
        nullable=True,
        doc='Автор административного изменения.',
        comment='Логин администратора, изменившего экспертные поля.',
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        doc='Дата создания отзыва.',
        comment='Момент сохранения анкеты.',
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
        doc='Дата изменения отзыва.',
        comment='Момент последнего изменения отзыва.',
    )

    chat_session: Mapped[ChatSession] = relationship(back_populates='feedback_entries', lazy='joined')

    def __repr__(self) -> str:
        return f"<SessionFeedback(id={self.id}, submission_id='{self.submission_id}')>"
