"""Convert legacy integer chat_messages.id primary key to UUID.

Ported from the pre-Alembic inline startup migration. No-op for databases
where the column is already UUID (including fresh installs from 0001).

Revision ID: 0002
Revises: 0001
Create Date: 2026-06-12

"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    data_type = bind.execute(
        sa.text(
            """
            SELECT data_type
            FROM information_schema.columns
            WHERE table_schema = 'public'
              AND table_name = 'chat_messages'
              AND column_name = 'id'
            """
        )
    ).scalar()
    if data_type not in {"bigint", "integer"}:
        return

    bind.execute(sa.text("CREATE EXTENSION IF NOT EXISTS pgcrypto"))
    bind.execute(sa.text("ALTER TABLE chat_messages ADD COLUMN IF NOT EXISTS id_uuid UUID"))
    bind.execute(sa.text("UPDATE chat_messages SET id_uuid = gen_random_uuid() WHERE id_uuid IS NULL"))
    bind.execute(sa.text("ALTER TABLE chat_messages ALTER COLUMN id_uuid SET NOT NULL"))
    bind.execute(sa.text("ALTER TABLE chat_messages DROP CONSTRAINT IF EXISTS chat_messages_pkey"))
    bind.execute(sa.text("ALTER TABLE chat_messages DROP COLUMN id"))
    bind.execute(sa.text("ALTER TABLE chat_messages RENAME COLUMN id_uuid TO id"))
    bind.execute(sa.text("ALTER TABLE chat_messages ADD PRIMARY KEY (id)"))


def downgrade() -> None:
    # Irreversible data migration (original integer ids are discarded on upgrade).
    pass
