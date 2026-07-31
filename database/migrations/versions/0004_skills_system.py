"""Multi-tenant agent skills system (base registry + tenant/user overlays).

Revision ID: 0004
Revises: 0003
Create Date: 2026-06-26

"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None


def _existing_tables() -> set[str]:
    return set(sa.inspect(op.get_bind()).get_table_names())


def upgrade() -> None:
    existing = _existing_tables()

    if "ai_skill_base" not in existing:
        op.create_table(
            "ai_skill_base",
            sa.Column("id", sa.String(128), primary_key=True),
            sa.Column("name", sa.String(255), nullable=False),
            sa.Column("description", sa.Text(), nullable=False),
            sa.Column("version", sa.Integer(), nullable=False, server_default="1"),
            sa.Column("enabled", sa.Boolean(), nullable=False, server_default=sa.true()),
            sa.Column(
                "created_at",
                sa.DateTime(timezone=True),
                nullable=False,
                server_default=sa.func.now(),
            ),
            sa.Column(
                "updated_at",
                sa.DateTime(timezone=True),
                nullable=False,
                server_default=sa.func.now(),
            ),
        )

    if "ai_skill_tenant_overlay" not in existing:
        op.create_table(
            "ai_skill_tenant_overlay",
            sa.Column(
                "id",
                postgresql.UUID(as_uuid=True),
                primary_key=True,
                server_default=sa.text("gen_random_uuid()"),
            ),
            sa.Column("tenant_id", sa.String(64), nullable=False),
            sa.Column(
                "skill_id",
                sa.String(128),
                sa.ForeignKey("ai_skill_base.id", ondelete="SET NULL"),
                nullable=True,
            ),
            sa.Column("name", sa.String(255), nullable=False),
            sa.Column("description", sa.Text(), nullable=False),
            sa.Column("overlay_type", sa.String(64), nullable=False),
            sa.Column("rule_text", sa.Text(), nullable=False),
            sa.Column("structured_rule", postgresql.JSONB(), nullable=True),
            sa.Column("status", sa.String(32), nullable=False, server_default="approved"),
            sa.Column("confidence", sa.Numeric(4, 3), nullable=True),
            sa.Column("source_trace_id", postgresql.UUID(as_uuid=True), nullable=True),
            sa.Column("created_by", sa.String(64), nullable=True),
            sa.Column("active", sa.Boolean(), nullable=False, server_default=sa.true()),
            sa.Column(
                "created_at",
                sa.DateTime(timezone=True),
                nullable=False,
                server_default=sa.func.now(),
            ),
            sa.Column(
                "updated_at",
                sa.DateTime(timezone=True),
                nullable=False,
                server_default=sa.func.now(),
            ),
            sa.UniqueConstraint("tenant_id", "name", name="uq_tenant_skill_name"),
        )
        op.create_index(
            "idx_tenant_overlay_lookup",
            "ai_skill_tenant_overlay",
            ["tenant_id", "skill_id", "status", "active"],
            if_not_exists=True,
        )

    if "ai_skill_user_overlay" not in existing:
        op.create_table(
            "ai_skill_user_overlay",
            sa.Column(
                "id",
                postgresql.UUID(as_uuid=True),
                primary_key=True,
                server_default=sa.text("gen_random_uuid()"),
            ),
            sa.Column("tenant_id", sa.String(64), nullable=False),
            sa.Column("user_id", sa.String(64), nullable=False),
            sa.Column(
                "skill_id",
                sa.String(128),
                sa.ForeignKey("ai_skill_base.id", ondelete="SET NULL"),
                nullable=True,
            ),
            sa.Column("name", sa.String(255), nullable=False),
            sa.Column("description", sa.Text(), nullable=False),
            sa.Column("overlay_type", sa.String(64), nullable=False),
            sa.Column("rule_text", sa.Text(), nullable=False),
            sa.Column("structured_rule", postgresql.JSONB(), nullable=True),
            sa.Column("status", sa.String(32), nullable=False, server_default="approved"),
            sa.Column("confidence", sa.Numeric(4, 3), nullable=True),
            sa.Column("source_trace_id", postgresql.UUID(as_uuid=True), nullable=True),
            sa.Column("active", sa.Boolean(), nullable=False, server_default=sa.true()),
            sa.Column(
                "created_at",
                sa.DateTime(timezone=True),
                nullable=False,
                server_default=sa.func.now(),
            ),
            sa.Column(
                "updated_at",
                sa.DateTime(timezone=True),
                nullable=False,
                server_default=sa.func.now(),
            ),
            sa.UniqueConstraint("tenant_id", "user_id", "name", name="uq_user_skill_name"),
        )
        op.create_index(
            "idx_user_overlay_lookup",
            "ai_skill_user_overlay",
            ["tenant_id", "user_id", "skill_id", "status", "active"],
            if_not_exists=True,
        )


def downgrade() -> None:
    op.drop_index("idx_user_overlay_lookup", table_name="ai_skill_user_overlay")
    op.drop_table("ai_skill_user_overlay")
    op.drop_index("idx_tenant_overlay_lookup", table_name="ai_skill_tenant_overlay")
    op.drop_table("ai_skill_tenant_overlay")
    op.drop_table("ai_skill_base")
