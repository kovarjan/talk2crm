"""Add rule_text to ai_skill_base and seed base skills from canonical content.

Revision ID: 0005
Revises: 0004
Create Date: 2026-07-02

"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op

from app.engine.base_skills import BASE_SKILLS

revision = "0005"
down_revision = "0004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    columns = {c["name"] for c in sa.inspect(bind).get_columns("ai_skill_base")}
    if "rule_text" not in columns:
        op.add_column(
            "ai_skill_base",
            sa.Column("rule_text", sa.Text(), nullable=False, server_default=""),
        )

    # Idempotent seed: insert missing base skills; only fill rule_text where it is
    # still empty so admin-edited content is never clobbered.
    for skill in BASE_SKILLS:
        bind.execute(
            sa.text(
                """
                INSERT INTO ai_skill_base (id, name, description, rule_text, version, enabled)
                VALUES (:id, :name, :description, :rule_text, :version, TRUE)
                ON CONFLICT (id) DO UPDATE
                    SET rule_text = EXCLUDED.rule_text
                    WHERE ai_skill_base.rule_text = ''
                """
            ),
            skill,
        )


def downgrade() -> None:
    op.drop_column("ai_skill_base", "rule_text")
