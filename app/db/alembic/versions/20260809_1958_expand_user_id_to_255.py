"""Расширяет идентификатор пользователя до 255 символов.

Revision ID: 20260809_1958
Revises: 20260729_1500
Create Date: 2026-08-09 19:58:00
"""

import sqlalchemy as sa
from alembic import op

revision = '20260809_1958'
down_revision = '20260729_1500'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.alter_column(
        'vera_chat_sessions',
        'user_id',
        existing_type=sa.String(length=100),
        type_=sa.String(length=255),
        existing_nullable=True,
    )
    op.alter_column(
        'vera_chat_turns',
        'user_id',
        existing_type=sa.String(length=100),
        type_=sa.String(length=255),
        existing_nullable=True,
    )


def downgrade() -> None:
    op.alter_column(
        'vera_chat_turns',
        'user_id',
        existing_type=sa.String(length=255),
        type_=sa.String(length=100),
        existing_nullable=True,
        postgresql_using='user_id::varchar(100)',
    )
    op.alter_column(
        'vera_chat_sessions',
        'user_id',
        existing_type=sa.String(length=255),
        type_=sa.String(length=100),
        existing_nullable=True,
        postgresql_using='user_id::varchar(100)',
    )
