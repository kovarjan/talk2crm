# Copyright 2026 Jan Kovář
# Copyright 2026 Acmark s.r.o.
# SPDX-License-Identifier: Apache-2.0
"""Widen chats.tool to hold a comma-separated capability list.

Revision ID: 0009
Revises: 0008
Create Date: 2026-09-15
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0009"
down_revision = "0008"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.alter_column("chats", "tool", type_=sa.String(128), existing_type=sa.String(32), existing_nullable=True)


def downgrade() -> None:
    op.alter_column("chats", "tool", type_=sa.String(32), existing_type=sa.String(128), existing_nullable=True)
