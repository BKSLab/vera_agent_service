"""Добавляет таблицу развёрнутых отзывов по сессии.

Revision ID: 20260729_1430
Revises: 20260729_1420
Create Date: 2026-07-29 14:30:00
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = '20260729_1430'
down_revision = '20260729_1420'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        'vera_session_feedback',
        sa.Column('id', sa.Uuid(), nullable=False, comment='Внутренний UUID развёрнутого отзыва.'),
        sa.Column('chat_session_id', sa.Uuid(), nullable=False, comment='Внешний ключ на сессию отзыва.'),
        sa.Column('submission_id', sa.String(length=100), nullable=False, comment='Идентификатор одной отправки анкеты.'),
        sa.Column('audience', sa.String(length=50), nullable=True, comment='Категория пользователя из анкеты.'),
        sa.Column('usefulness', sa.Integer(), nullable=True, comment='Оценка полезности от 1 до 5.'),
        sa.Column('trust', sa.Integer(), nullable=True, comment='Оценка доверия от 1 до 5.'),
        sa.Column('comment', sa.Text(), nullable=True, comment='Свободный комментарий пользователя.'),
        sa.Column('contact_email', sa.String(length=320), nullable=True, comment='Добровольно указанный контактный email.'),
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
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False, comment='Момент сохранения анкеты.'),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False, comment='Момент последнего изменения отзыва.'),
        sa.CheckConstraint('usefulness IS NULL OR usefulness BETWEEN 1 AND 5', name='ck_vera_session_feedback_usefulness'),
        sa.CheckConstraint('trust IS NULL OR trust BETWEEN 1 AND 5', name='ck_vera_session_feedback_trust'),
        sa.ForeignKeyConstraint(['chat_session_id'], ['vera_chat_sessions.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('submission_id', name='uq_vera_session_feedback_submission_id'),
    )
    op.create_index('ix_vera_session_feedback_chat_session_id', 'vera_session_feedback', ['chat_session_id'])
    op.create_index('ix_vera_session_feedback_submission_id', 'vera_session_feedback', ['submission_id'])
    op.create_index('ix_vera_session_feedback_review_status', 'vera_session_feedback', ['review_status'])


def downgrade() -> None:
    op.drop_index('ix_vera_session_feedback_review_status', table_name='vera_session_feedback')
    op.drop_index('ix_vera_session_feedback_submission_id', table_name='vera_session_feedback')
    op.drop_index('ix_vera_session_feedback_chat_session_id', table_name='vera_session_feedback')
    op.drop_table('vera_session_feedback')
