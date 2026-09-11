"""Add tool and pinned columns to chats.

Revision ID: 0008
Revises: 0007
Create Date: 2026-09-11

"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0008"
down_revision = "0007"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    columns = {c["name"] for c in sa.inspect(bind).get_columns("chats")}
    if "tool" not in columns:
        op.add_column("chats", sa.Column("tool", sa.String(length=32), nullable=True))
    if "pinned" not in columns:
        op.add_column(
            "chats",
            sa.Column("pinned", sa.Boolean(), nullable=False, server_default=sa.false()),
        )


def downgrade() -> None:
    op.drop_column("chats", "pinned")
    op.drop_column("chats", "tool")
