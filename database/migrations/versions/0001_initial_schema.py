"""Initial schema (tenants, chats, chat_messages).

Idempotent: deployments that predate Alembic already have these tables from
the old ``Base.metadata.create_all`` startup path, so each table is only
created when missing.

Revision ID: 0001
Revises:
Create Date: 2026-06-12

"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def _existing_tables() -> set[str]:
    return set(sa.inspect(op.get_bind()).get_table_names())


def upgrade() -> None:
    existing = _existing_tables()

    if "tenants" not in existing:
        op.create_table(
            "tenants",
            sa.Column("id", sa.String(100), primary_key=True),
            sa.Column("name", sa.String(255), nullable=False),
            sa.Column("crm_base_url", sa.String(1024), nullable=False),
            sa.Column("crm_token_encrypted", sa.Text(), nullable=False),
            sa.Column("active", sa.Boolean(), nullable=False, server_default=sa.true()),
            sa.Column("metadata_json", sa.JSON(), nullable=False, server_default=sa.text("'{}'")),
            sa.Column(
                "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
            ),
            sa.Column(
                "updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
            ),
        )

    if "chats" not in existing:
        op.create_table(
            "chats",
            sa.Column("id", sa.String(64), primary_key=True),
            sa.Column(
                "tenant_id",
                sa.String(100),
                sa.ForeignKey("tenants.id", ondelete="CASCADE"),
                nullable=False,
            ),
            sa.Column("user_id", sa.String(255), nullable=False, index=True),
            sa.Column("name", sa.String(255), nullable=True),
            sa.Column(
                "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
            ),
            sa.Column(
                "updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
            ),
            sa.UniqueConstraint("id", "tenant_id", name="uq_chat_tenant"),
        )

    if "chat_messages" not in existing:
        op.create_table(
            "chat_messages",
            sa.Column("id", sa.Uuid(as_uuid=False), primary_key=True),
            sa.Column(
                "chat_id",
                sa.String(64),
                sa.ForeignKey("chats.id", ondelete="CASCADE"),
                nullable=False,
                index=True,
            ),
            sa.Column("tenant_id", sa.String(100), nullable=False, index=True),
            sa.Column("user_id", sa.String(255), nullable=False, index=True),
            sa.Column("role", sa.String(20), nullable=False),
            sa.Column("content", sa.Text(), nullable=False),
            sa.Column("metadata_json", sa.JSON(), nullable=False, server_default=sa.text("'{}'")),
            sa.Column(
                "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
            ),
        )


def downgrade() -> None:
    op.drop_table("chat_messages")
    op.drop_table("chats")
    op.drop_table("tenants")
