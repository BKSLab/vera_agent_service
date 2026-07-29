"""Добавляет полнотекстовый поиск по комментариям к сессиям.

Revision ID: 20260729_1450
Revises: 20260729_1440
Create Date: 2026-07-29 14:50:00
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = '20260729_1450'
down_revision = '20260729_1440'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        'vera_session_feedback',
        sa.Column(
            'search_vector',
            postgresql.TSVECTOR(),
            sa.Computed(
                "to_tsvector('russian'::regconfig, coalesce(comment, ''))",
                persisted=True,
            ),
            nullable=False,
            comment='Автоматически формируемый tsvector для русскоязычного поиска.',
        ),
    )
    op.create_index(
        'ix_vera_session_feedback_search_vector',
        'vera_session_feedback',
        ['search_vector'],
        postgresql_using='gin',
    )


def downgrade() -> None:
    op.drop_index(
        'ix_vera_session_feedback_search_vector',
        table_name='vera_session_feedback',
    )
    op.drop_column('vera_session_feedback', 'search_vector')
