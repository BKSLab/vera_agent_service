"""Добавляет состояние доставки реплики: аренда, попытки и точный исход.

Revision ID: 20260809_2110
Revises: 20260809_1958
Create Date: 2026-08-09 21:10:00
"""

import sqlalchemy as sa
from alembic import op

revision = '20260809_2110'
down_revision = '20260809_1958'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        'vera_chat_turns',
        sa.Column(
            'terminal_detail',
            sa.Text(),
            nullable=True,
            comment='Пользовательский текст ошибки, отправленный в терминальном SSE-событии.',
        ),
    )
    op.add_column(
        'vera_chat_turns',
        sa.Column(
            'lease_until',
            sa.DateTime(timezone=True),
            nullable=True,
            comment='До этого момента реплика считается обрабатываемой живым worker.',
        ),
    )
    op.add_column(
        'vera_chat_turns',
        sa.Column(
            'attempt_count',
            sa.Integer(),
            nullable=False,
            server_default='0',
            comment='Сколько раз реплика бралась в обработку, включая повторные захваты.',
        ),
    )
    op.add_column(
        'vera_chat_turns',
        sa.Column(
            'worker_id',
            sa.String(length=100),
            nullable=True,
            comment='Идентификатор процесса, удерживающего текущую аренду.',
        ),
    )

    # Прежний обобщённый `failed` означал ровно один случай: ответ не был
    # сформирован и ни один токен не ушёл пользователю. Это `generation_failed`
    # в новой номенклатуре — данные переносятся, а не теряются.
    op.execute(
        "UPDATE vera_chat_turns SET status = 'generation_failed' WHERE status = 'failed'"
    )

    op.create_index(
        'ix_vera_chat_turns_status_lease_until',
        'vera_chat_turns',
        ['status', 'lease_until'],
    )


def downgrade() -> None:
    op.drop_index('ix_vera_chat_turns_status_lease_until', table_name='vera_chat_turns')

    # Обратный перенос: все новые терминальные статусы неуспеха схлопываются
    # в прежний `failed`, который был единственным доступным значением.
    op.execute(
        "UPDATE vera_chat_turns SET status = 'failed' "
        "WHERE status IN ('generation_failed', 'stream_interrupted', "
        "'delivery_unconfirmed', 'cancelled')"
    )

    op.drop_column('vera_chat_turns', 'worker_id')
    op.drop_column('vera_chat_turns', 'attempt_count')
    op.drop_column('vera_chat_turns', 'lease_until')
    op.drop_column('vera_chat_turns', 'terminal_detail')
