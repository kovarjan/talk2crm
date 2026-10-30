# Copyright 2026 Jan Kovář
# Copyright 2026 Acmark s.r.o.
# SPDX-License-Identifier: Apache-2.0
"""Optional per-tenant LLM temperature override (NULL = code default).

Revision ID: 0015
Revises: 0014
Create Date: 2026-10-30
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0015"
down_revision = "0014"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("tenants", sa.Column("llm_temperature", sa.Float(), nullable=True))


def downgrade() -> None:
    op.drop_column("tenants", "llm_temperature")
