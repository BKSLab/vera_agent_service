from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

from sqlalchemy import DateTime, ForeignKey, String, Text, UniqueConstraint, Uuid, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.models.base import Base

if TYPE_CHECKING:
    from app.db.models.chat_turn import ChatTurn


class MessageFeedback(Base):
    """Пользовательская оценка одного ответа Веры."""

    __tablename__ = 'vera_message_feedback'
    __table_args__ = (
        UniqueConstraint(
            'chat_turn_id',
            name='uq_vera_message_feedback_chat_turn_id',
        ),
    )

    id: Mapped[UUID] = mapped_column(
        Uuid,
        primary_key=True,
        default=uuid4,
        doc='Внутренний идентификатор оценки.',
        comment='Внутренний UUID оценки ответа.',
    )
    chat_turn_id: Mapped[UUID] = mapped_column(
        ForeignKey('vera_chat_turns.id', ondelete='CASCADE'),
        nullable=False,
        index=True,
        doc='Ссылка на оценённую реплику.',
        comment='Уникальный внешний ключ на оценённый ответ.',
    )
    value: Mapped[str] = mapped_column(
        String(length=10),
        nullable=False,
        doc='Значение оценки.',
        comment='Пользовательская оценка up или down.',
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
        doc='Дата создания оценки.',
        comment='Момент первой оценки ответа.',
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
        doc='Дата изменения оценки.',
        comment='Момент последнего изменения оценки.',
    )

    chat_turn: Mapped[ChatTurn] = relationship(back_populates='feedback', lazy='joined')

    def __repr__(self) -> str:
        return f"<MessageFeedback(id={self.id}, value='{self.value}', review_status='{self.review_status}')>"
