"""Добавляет таблицу постоянных сессий диалога.

Revision ID: 20260729_1400
Revises:
Create Date: 2026-07-29 14:00:00
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = '20260729_1400'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        'vera_chat_sessions',
        sa.Column('id', sa.Uuid(), nullable=False, comment='Внутренний UUID сессии диалога.'),
        sa.Column('session_id', sa.String(length=100), nullable=False, comment='Идентификатор сессии, поступивший от сайта.'),
        sa.Column('user_id', sa.String(length=100), nullable=True, comment='Идентификатор пользователя сайта, если он был авторизован.'),
        sa.Column(
            'service_metadata',
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
            comment='Служебные метаданные без текстов диалога и персональных данных.',
        ),
        sa.Column(
            'created_at',
            sa.DateTime(timezone=True),
            server_default=sa.text('now()'),
            nullable=False,
            comment='Момент создания постоянной записи сессии.',
        ),
        sa.Column(
            'last_activity_at',
            sa.DateTime(timezone=True),
            server_default=sa.text('now()'),
            nullable=False,
            comment='Момент последнего сохранённого запроса в сессии.',
        ),
        sa.Column('closed_at', sa.DateTime(timezone=True), nullable=True, comment='Момент явного закрытия сессии.'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('session_id', name='uq_vera_chat_sessions_session_id'),
    )
    op.create_index('ix_vera_chat_sessions_session_id', 'vera_chat_sessions', ['session_id'])
    op.create_index('ix_vera_chat_sessions_user_id', 'vera_chat_sessions', ['user_id'])
    op.create_index('ix_vera_chat_sessions_last_activity_at', 'vera_chat_sessions', ['last_activity_at'])


def downgrade() -> None:
    op.drop_index('ix_vera_chat_sessions_last_activity_at', table_name='vera_chat_sessions')
    op.drop_index('ix_vera_chat_sessions_user_id', table_name='vera_chat_sessions')
    op.drop_index('ix_vera_chat_sessions_session_id', table_name='vera_chat_sessions')
    op.drop_table('vera_chat_sessions')
