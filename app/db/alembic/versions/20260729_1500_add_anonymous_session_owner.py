"""Добавляет владельца анонимной сессии.

Revision ID: 20260729_1500
Revises: 20260729_1450
Create Date: 2026-07-29 15:00:00
"""

import sqlalchemy as sa
from alembic import op

revision = '20260729_1500'
down_revision = '20260729_1450'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        'vera_chat_sessions',
        sa.Column(
            'anonymous_token_hash',
            sa.String(length=64),
            nullable=True,
            comment='Хеш серверного токена владельца анонимной сессии.',
        ),
    )
    op.create_index(
        'ix_vera_chat_sessions_anonymous_token_hash',
        'vera_chat_sessions',
        ['anonymous_token_hash'],
    )


def downgrade() -> None:
    op.drop_index(
        'ix_vera_chat_sessions_anonymous_token_hash',
        table_name='vera_chat_sessions',
    )
    op.drop_column('vera_chat_sessions', 'anonymous_token_hash')
