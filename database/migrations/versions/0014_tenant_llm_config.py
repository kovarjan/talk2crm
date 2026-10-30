# Copyright 2026 Jan Kovář
# Copyright 2026 Acmark s.r.o.
# SPDX-License-Identifier: Apache-2.0
"""Optional per-tenant LLM endpoint, key and models (NULL = system default).

Revision ID: 0014
Revises: 0013
Create Date: 2026-10-30
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0014"
down_revision = "0013"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("tenants", sa.Column("llm_base_url", sa.String(1024), nullable=True))
    op.add_column("tenants", sa.Column("llm_api_key", sa.Text(), nullable=True))
    op.add_column("tenants", sa.Column("llm_model", sa.String(255), nullable=True))
    op.add_column("tenants", sa.Column("llm_title_model", sa.String(255), nullable=True))


def downgrade() -> None:
    for column in ("llm_title_model", "llm_model", "llm_api_key", "llm_base_url"):
        op.drop_column("tenants", column)
