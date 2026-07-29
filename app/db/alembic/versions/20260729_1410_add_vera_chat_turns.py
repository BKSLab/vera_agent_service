"""Добавляет таблицу постоянных реплик диалога.

Revision ID: 20260729_1410
Revises: 20260729_1400
Create Date: 2026-07-29 14:10:00
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = '20260729_1410'
down_revision = '20260729_1400'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        'vera_chat_turns',
        sa.Column('id', sa.Uuid(), nullable=False, comment='Внутренний UUID реплики диалога.'),
        sa.Column('request_id', sa.String(length=100), nullable=False, comment='Идентификатор RabbitMQ-запроса и SSE-канала.'),
        sa.Column('chat_session_id', sa.Uuid(), nullable=False, comment='Внешний ключ на постоянную запись сессии.'),
        sa.Column('sequence_number', sa.Integer(), nullable=False, comment='Порядковый номер вопроса и ответа внутри сессии.'),
        sa.Column('user_id', sa.String(length=100), nullable=True, comment='Идентификатор пользователя сайта, если он был авторизован.'),
        sa.Column('question', sa.Text(), nullable=False, comment='Исходный текст запроса, полученный из RabbitMQ.'),
        sa.Column('answer', sa.Text(), nullable=True, comment='Конкатенация токенов финального ответа, отправленных через SSE.'),
        sa.Column(
            'sources',
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'[]'::jsonb"),
            nullable=False,
            comment='Структурированный список источников, использованных при ответе.',
        ),
        sa.Column(
            'technical_metadata',
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
            comment='Маршрут обработки и имена инструментов без секретов и текстов.',
        ),
        sa.Column('status', sa.String(length=30), server_default='processing', nullable=False, comment='Статус обработки реплики.'),
        sa.Column('safe_error', sa.Text(), nullable=True, comment='Безопасное описание ошибки без секретов.'),
        sa.Column('started_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False, comment='Момент начала обработки.'),
        sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True, comment='Момент завершения обработки.'),
        sa.Column('latency_ms', sa.Integer(), nullable=True, comment='Полная длительность обработки в миллисекундах.'),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False, comment='Момент создания записи.'),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False, comment='Момент последнего изменения записи.'),
        sa.ForeignKeyConstraint(['chat_session_id'], ['vera_chat_sessions.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('chat_session_id', 'sequence_number', name='uq_vera_chat_turns_session_sequence'),
        sa.UniqueConstraint('request_id', name='uq_vera_chat_turns_request_id'),
    )
    op.create_index('ix_vera_chat_turns_request_id', 'vera_chat_turns', ['request_id'])
    op.create_index('ix_vera_chat_turns_chat_session_id', 'vera_chat_turns', ['chat_session_id'])
    op.create_index('ix_vera_chat_turns_session_sequence', 'vera_chat_turns', ['chat_session_id', 'sequence_number'])
    op.create_index('ix_vera_chat_turns_status_created_at', 'vera_chat_turns', ['status', 'created_at'])


def downgrade() -> None:
    op.drop_index('ix_vera_chat_turns_status_created_at', table_name='vera_chat_turns')
    op.drop_index('ix_vera_chat_turns_session_sequence', table_name='vera_chat_turns')
    op.drop_index('ix_vera_chat_turns_chat_session_id', table_name='vera_chat_turns')
    op.drop_index('ix_vera_chat_turns_request_id', table_name='vera_chat_turns')
    op.drop_table('vera_chat_turns')
