from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

from sqlalchemy import (
    Computed,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    Uuid,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.models.base import Base

if TYPE_CHECKING:
    from app.db.models.chat_session import ChatSession
    from app.db.models.message_feedback import MessageFeedback


class ChatTurn(Base):
    """Постоянная запись одного вопроса пользователя и ответа Веры."""

    __tablename__ = 'vera_chat_turns'
    __table_args__ = (
        UniqueConstraint(
            'chat_session_id',
            'sequence_number',
            name='uq_vera_chat_turns_session_sequence',
        ),
        UniqueConstraint('request_id', name='uq_vera_chat_turns_request_id'),
        Index('ix_vera_chat_turns_session_sequence', 'chat_session_id', 'sequence_number'),
        Index('ix_vera_chat_turns_status_created_at', 'status', 'created_at'),
        Index(
            'ix_vera_chat_turns_search_vector',
            'search_vector',
            postgresql_using='gin',
        ),
    )

    id: Mapped[UUID] = mapped_column(
        Uuid,
        primary_key=True,
        default=uuid4,
        doc='Внутренний идентификатор реплики.',
        comment='Внутренний UUID реплики диалога.',
    )
    request_id: Mapped[str] = mapped_column(
        String(length=100),
        nullable=False,
        index=True,
        doc='Внешний идентификатор запроса.',
        comment='Идентификатор RabbitMQ-запроса и SSE-канала.',
    )
    chat_session_id: Mapped[UUID] = mapped_column(
        ForeignKey('vera_chat_sessions.id', ondelete='CASCADE'),
        nullable=False,
        index=True,
        doc='Ссылка на сессию.',
        comment='Внешний ключ на постоянную запись сессии.',
    )
    sequence_number: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        doc='Порядковый номер реплики.',
        comment='Порядковый номер вопроса и ответа внутри сессии.',
    )
    user_id: Mapped[str | None] = mapped_column(
        String(length=100),
        nullable=True,
        doc='Идентификатор пользователя.',
        comment='Идентификатор пользователя сайта, если он был авторизован.',
    )
    question: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        doc='Точный вопрос пользователя.',
        comment='Исходный текст запроса, полученный из RabbitMQ.',
    )
    answer: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        doc='Фактически отправленный ответ.',
        comment='Конкатенация токенов финального ответа, отправленных через SSE.',
    )
    search_vector: Mapped[str] = mapped_column(
        TSVECTOR,
        Computed(
            "setweight(to_tsvector('russian'::regconfig, coalesce(question, '')), 'A') "
            "|| setweight(to_tsvector('russian'::regconfig, coalesce(answer, '')), 'B')",
            persisted=True,
        ),
        nullable=False,
        deferred=True,
        doc='Полнотекстовый индекс вопроса и ответа.',
        comment='Автоматически формируемый tsvector для русскоязычного поиска.',
    )
    sources: Mapped[list] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
        doc='Источники ответа.',
        comment='Структурированный список источников, использованных при ответе.',
    )
    technical_metadata: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        doc='Безопасные технические метаданные.',
        comment='Маршрут обработки и имена инструментов без секретов и текстов.',
    )
    status: Mapped[str] = mapped_column(
        String(length=30),
        nullable=False,
        default='processing',
        doc='Статус обработки реплики.',
        comment='Статус обработки реплики.',
    )
    safe_error: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        doc='Безопасное описание ошибки.',
        comment='Безопасное описание ошибки без секретов.',
    )
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        doc='Дата начала обработки.',
        comment='Момент начала обработки.',
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        doc='Дата завершения обработки.',
        comment='Момент завершения обработки.',
    )
    latency_ms: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
        doc='Длительность обработки.',
        comment='Полная длительность обработки в миллисекундах.',
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        doc='Дата создания записи.',
        comment='Момент создания записи.',
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
        doc='Дата последнего обновления.',
        comment='Момент последнего изменения записи.',
    )

    chat_session: Mapped[ChatSession] = relationship(back_populates='turns', lazy='joined')
    feedback: Mapped[MessageFeedback | None] = relationship(
        back_populates='chat_turn',
        cascade='all, delete-orphan',
        uselist=False,
        lazy='raise',
    )

    def __repr__(self) -> str:
        return f"<ChatTurn(id={self.id}, request_id='{self.request_id}', status='{self.status}')>"
