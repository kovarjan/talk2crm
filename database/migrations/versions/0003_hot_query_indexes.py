"""Composite indexes for the hot chat queries.

chat_messages is always filtered by chat_id and ordered by created_at;
chats listing filters (tenant_id, user_id) ordered by updated_at.

Revision ID: 0003
Revises: 0002
Create Date: 2026-06-12

"""
from __future__ import annotations

from alembic import op

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_index(
        "ix_chat_messages_chat_created",
        "chat_messages",
        ["chat_id", "created_at"],
        if_not_exists=True,
    )
    op.create_index(
        "ix_chats_tenant_user_updated",
        "chats",
        ["tenant_id", "user_id", "updated_at"],
        if_not_exists=True,
    )


def downgrade() -> None:
    op.drop_index("ix_chat_messages_chat_created", table_name="chat_messages")
    op.drop_index("ix_chats_tenant_user_updated", table_name="chats")
