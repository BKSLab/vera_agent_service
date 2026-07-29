"""Добавляет таблицу оценок ответов.

Revision ID: 20260729_1420
Revises: 20260729_1410
Create Date: 2026-07-29 14:20:00
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = '20260729_1420'
down_revision = '20260729_1410'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        'vera_message_feedback',
        sa.Column('id', sa.Uuid(), nullable=False, comment='Внутренний UUID оценки ответа.'),
        sa.Column('chat_turn_id', sa.Uuid(), nullable=False, comment='Уникальный внешний ключ на оценённый ответ.'),
        sa.Column('value', sa.String(length=10), nullable=False, comment='Пользовательская оценка up или down.'),
        sa.Column('review_status', sa.String(length=20), server_default='new', nullable=False, comment='Статус экспертной проверки.'),
        sa.Column('expert_note', sa.Text(), nullable=True, comment='Служебная заметка, не доступная пользователю.'),
        sa.Column(
            'tags',
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'[]'::jsonb"),
            nullable=False,
            comment='Список причин или категорий экспертной проверки.',
        ),
        sa.Column('updated_by_admin', sa.String(length=100), nullable=True, comment='Логин администратора, изменившего экспертные поля.'),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False, comment='Момент первой оценки ответа.'),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False, comment='Момент последнего изменения оценки.'),
        sa.ForeignKeyConstraint(['chat_turn_id'], ['vera_chat_turns.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('chat_turn_id', name='uq_vera_message_feedback_chat_turn_id'),
    )
    op.create_index('ix_vera_message_feedback_chat_turn_id', 'vera_message_feedback', ['chat_turn_id'])
    op.create_index('ix_vera_message_feedback_review_status', 'vera_message_feedback', ['review_status'])


def downgrade() -> None:
    op.drop_index('ix_vera_message_feedback_review_status', table_name='vera_message_feedback')
    op.drop_index('ix_vera_message_feedback_chat_turn_id', table_name='vera_message_feedback')
    op.drop_table('vera_message_feedback')
