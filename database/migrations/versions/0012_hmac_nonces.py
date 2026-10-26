# Copyright 2026 Jan Kovář
# Copyright 2026 Acmark s.r.o.
# SPDX-License-Identifier: Apache-2.0
"""Shared HMAC nonce store for replay protection across workers.

Revision ID: 0012
Revises: 0011
Create Date: 2026-10-21
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0012"
down_revision = "0011"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "hmac_nonces",
        sa.Column("key_id", sa.String(200), primary_key=True),
        sa.Column("nonce", sa.String(200), primary_key=True),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_hmac_nonces_expires_at", "hmac_nonces", ["expires_at"])


def downgrade() -> None:
    op.drop_index("ix_hmac_nonces_expires_at", table_name="hmac_nonces")
    op.drop_table("hmac_nonces")
