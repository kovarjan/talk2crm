"""
Pydantic contracts for the multi-tenant agent skills system.

Hierarchy:
  base skill (filesystem SKILL.md)
    └── tenant overlay  (DB: ai_skill_tenant_overlay)
         └── user overlay  (DB: ai_skill_user_overlay)

Scope precedence for runtime injection:
  session > user > tenant > base
  BUT: business_rule/field_mapping/safety_rule tenant overlays
       override user response_preference overlays.
"""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

OverlayType = Literal[
    "business_rule",       # → always tenant scope
    "field_mapping",       # → always tenant scope
    "entity_alias",        # → tenant scope
    "workflow_default",    # → user scope
    "response_preference", # → always user scope
    "tool_usage_rule",     # → tenant scope
    "safety_rule",         # → tenant scope
]

# Deterministic scope decision — never ask LLM to guess scope
SCOPE_BY_OVERLAY_TYPE: dict[str, Literal["tenant", "user"]] = {
    "business_rule":       "tenant",
    "field_mapping":       "tenant",
    "entity_alias":        "tenant",
    "workflow_default":    "user",
    "response_preference": "user",
    "tool_usage_rule":     "tenant",
    "safety_rule":         "tenant",
}


# ---------------------------------------------------------------------------
# Base skill (read-only at runtime — lives in filesystem)
# ---------------------------------------------------------------------------

class BaseSkillInfo(BaseModel):
    id: str
    name: str
    description: str
    rule_text: str = ""
    version: int = 1
    enabled: bool = True

    model_config = {"from_attributes": True}


# ---------------------------------------------------------------------------
# Skill index entry — name + description only, minimal context footprint
# ---------------------------------------------------------------------------

class SkillIndexEntry(BaseModel):
    name: str
    description: str
    scope: Literal["base", "tenant", "user"]
    overlay_type: Optional[OverlayType] = None


# ---------------------------------------------------------------------------
# Full skill overlay — returned when agent requests full content
# ---------------------------------------------------------------------------

class TenantOverlay(BaseModel):
    id: uuid.UUID
    tenant_id: str
    skill_id: Optional[str]
    name: str
    description: str
    overlay_type: OverlayType
    rule_text: str
    structured_rule: Optional[dict[str, Any]]
    confidence: Optional[float]
    source_trace_id: Optional[uuid.UUID]
    created_by: Optional[str]
    active: bool
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class UserOverlay(BaseModel):
    id: uuid.UUID
    tenant_id: str
    user_id: str
    skill_id: Optional[str]
    name: str
    description: str
    overlay_type: OverlayType
    rule_text: str
    structured_rule: Optional[dict[str, Any]]
    confidence: Optional[float]
    source_trace_id: Optional[uuid.UUID]
    active: bool
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


# ---------------------------------------------------------------------------
# Create / upsert inputs
# ---------------------------------------------------------------------------

class TenantOverlayCreate(BaseModel):
    name: str = Field(max_length=255)
    description: str = Field(
        description="One line, used in skill index for relevance matching."
    )
    overlay_type: OverlayType
    rule_text: str
    skill_id: Optional[str] = None
    structured_rule: Optional[dict[str, Any]] = None
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    source_trace_id: Optional[uuid.UUID] = None
    created_by: Optional[str] = None


class UserOverlayCreate(BaseModel):
    name: str = Field(max_length=255)
    description: str
    overlay_type: OverlayType
    rule_text: str
    skill_id: Optional[str] = None
    structured_rule: Optional[dict[str, Any]] = None
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    source_trace_id: Optional[uuid.UUID] = None


# ---------------------------------------------------------------------------
# Assembled runtime skill context — what gets injected into system prompt
# ---------------------------------------------------------------------------

class RuntimeSkillContext(BaseModel):
    """
    Final merged context for a single request.
    Built by SkillService.assemble_context().
    """
    tenant_id: str
    user_id: str
    base_skills: list[BaseSkillInfo] = Field(default_factory=list)
    tenant_overlays: list[TenantOverlay]
    user_overlays: list[UserOverlay]

    def to_prompt_block(self) -> str:
        """Renders the skill context as a Czech prompt block for system prompt injection."""
        if not self.base_skills and not self.tenant_overlays and not self.user_overlays:
            return ""

        lines = ["## Pravidla a dovednosti\n"]

        if self.base_skills:
            lines.append("### Základní pravidla")
            for s in self.base_skills:
                lines.append(f"**{s.name}**")
                lines.append(s.rule_text)
            lines.append("")

        if self.tenant_overlays:
            lines.append("### Pravidla organizace")
            for o in self.tenant_overlays:
                lines.append(f"- **{o.name}**: {o.rule_text}")
            lines.append("")

        if self.user_overlays:
            lines.append("### Vaše osobní preference")
            for o in self.user_overlays:
                lines.append(f"- **{o.name}**: {o.rule_text}")
            lines.append("")

        lines.append(
            "_Tato pravidla mají přednost před výchozím chováním. "
            "Pravidla organizace a osobní preference mají přednost před základními pravidly; "
            "Pravidla organizace mají přednost před osobními preferencemi "
            "v případě konfliktu._"
        )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Correction classifier output
# ---------------------------------------------------------------------------

class SkillCorrectionCandidate(BaseModel):
    """
    Output of SkillCaptureService._distill().
    LLM classifies overlay_type; code derives scope deterministically.
    """
    name: str
    description: str
    overlay_type: OverlayType
    rule_text: str
    structured_rule: Optional[dict[str, Any]] = None
    confidence: float = Field(ge=0.0, le=1.0)
    target_skill_id: Optional[str] = None

    @property
    def recommended_scope(self) -> Literal["tenant", "user"]:
        return SCOPE_BY_OVERLAY_TYPE[self.overlay_type]
