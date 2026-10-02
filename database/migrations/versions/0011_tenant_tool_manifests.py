# Copyright 2026 Jan Kovář
# Copyright 2026 Acmark s.r.o.
# SPDX-License-Identifier: Apache-2.0
"""Last good CRM tool manifest per tenant (AI tool registry fallback).

Revision ID: 0011
Revises: 0010
Create Date: 2026-09-25
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0011"
down_revision = "0010"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "tenant_tool_manifests",
        sa.Column("tenant_id", sa.String(100), sa.ForeignKey("tenants.id", ondelete="CASCADE"), primary_key=True),
        sa.Column("manifest_hash", sa.String(80), nullable=False),
        sa.Column("manifest_json", sa.JSON(), nullable=False),
        sa.Column("fetched_at", sa.DateTime(timezone=True), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("tenant_tool_manifests")
