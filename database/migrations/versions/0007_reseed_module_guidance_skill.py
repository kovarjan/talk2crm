# Copyright 2026 Jan Kovář
# Copyright 2026 Acmark s.r.o.
# SPDX-License-Identifier: Apache-2.0

"""Version-aware reseed of base skills (crm-module-field-guidance v2: orders + line items).

Inserts missing base skills and updates rule_text/name/description only when the
canonical version in app/engine/base_skills.py is newer than the DB row, so
admin edits made on top of the current version are preserved unless the
canonical content moves forward.

Revision ID: 0007
Revises: 0006
Create Date: 2026-07-14

"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op

from app.engine.base_skills import BASE_SKILLS

revision = "0007"
down_revision = "0006"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    for skill in BASE_SKILLS:
        bind.execute(
            sa.text(
                """
                INSERT INTO ai_skill_base (id, name, description, rule_text, version, enabled)
                VALUES (:id, :name, :description, :rule_text, :version, TRUE)
                ON CONFLICT (id) DO UPDATE
                    SET name = EXCLUDED.name,
                        description = EXCLUDED.description,
                        rule_text = EXCLUDED.rule_text,
                        version = EXCLUDED.version
                    WHERE ai_skill_base.version < EXCLUDED.version
                       OR ai_skill_base.rule_text = ''
                """
            ),
            skill,
        )


def downgrade() -> None:
    pass  # content-only reseed; nothing structural to revert
