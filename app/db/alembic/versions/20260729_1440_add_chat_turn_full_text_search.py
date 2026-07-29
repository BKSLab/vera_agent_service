"""Добавляет полнотекстовый поиск по вопросам и ответам.

Revision ID: 20260729_1440
Revises: 20260729_1430
Create Date: 2026-07-29 14:40:00
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = '20260729_1440'
down_revision = '20260729_1430'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        'vera_chat_turns',
        sa.Column(
            'search_vector',
            postgresql.TSVECTOR(),
            sa.Computed(
                "setweight(to_tsvector('russian'::regconfig, coalesce(question, '')), 'A') "
                "|| setweight(to_tsvector('russian'::regconfig, coalesce(answer, '')), 'B')",
                persisted=True,
            ),
            nullable=False,
            comment='Автоматически формируемый tsvector для русскоязычного поиска.',
        ),
    )
    op.create_index(
        'ix_vera_chat_turns_search_vector',
        'vera_chat_turns',
        ['search_vector'],
        postgresql_using='gin',
    )


def downgrade() -> None:
    op.drop_index('ix_vera_chat_turns_search_vector', table_name='vera_chat_turns')
    op.drop_column('vera_chat_turns', 'search_vector')
