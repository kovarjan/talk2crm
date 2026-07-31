"""
SkillService — DB-backed skill CRUD and context assembly.

Responsibilities:
- Load skill index (name + description) for a tenant/user pair
- Load full overlay content for injection
- Upsert tenant and user overlays
- Respect skills_modification_enabled flag
- Never cross tenant boundaries

No LLM calls in this file. See SkillCaptureService for correction distillation.
"""
from __future__ import annotations

import logging
import uuid
from typing import Optional

from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings
from app.domain.skill_contracts import (
    BaseSkillInfo,
    RuntimeSkillContext,
    SkillIndexEntry,
    TenantOverlay,
    TenantOverlayCreate,
    UserOverlay,
    UserOverlayCreate,
)
from database.skill_models import AISkillBase, AISkillTenantOverlay, AISkillUserOverlay

logger = logging.getLogger(__name__)

MAX_RULE_TEXT_CHARS = 2000


def _coerce_trace_id(
    data_trace_id: Optional[uuid.UUID], request_id: Optional[str]
) -> Optional[uuid.UUID]:
    """source_trace_id is a UUID column — never store a raw string."""
    if data_trace_id is not None:
        return data_trace_id
    if not request_id:
        return None
    try:
        return uuid.UUID(str(request_id))
    except (ValueError, AttributeError):
        return None


class SkillService:

    def __init__(self, db: AsyncSession, settings: Settings) -> None:
        self._db = db
        self._settings = settings

    # ------------------------------------------------------------------
    # Index loading (name + description only — minimal context cost)
    # ------------------------------------------------------------------

    async def load_index(
        self, tenant_id: str, user_id: str
    ) -> list[SkillIndexEntry]:
        """
        Returns skill index for a tenant/user pair.
        User overlay with same name as tenant overlay wins (dedup by name).
        """
        if not self._settings.skills_enabled:
            return []

        max_skills = self._settings.skills_max_per_context
        tenant_rows = await self._db.execute(
            select(
                AISkillTenantOverlay.name,
                AISkillTenantOverlay.description,
                AISkillTenantOverlay.overlay_type,
            ).where(
                and_(
                    AISkillTenantOverlay.tenant_id == tenant_id,
                    AISkillTenantOverlay.active == True,  # noqa: E712
                    AISkillTenantOverlay.status == "approved",
                )
            ).limit(max_skills)
        )

        user_rows = await self._db.execute(
            select(
                AISkillUserOverlay.name,
                AISkillUserOverlay.description,
                AISkillUserOverlay.overlay_type,
            ).where(
                and_(
                    AISkillUserOverlay.tenant_id == tenant_id,
                    AISkillUserOverlay.user_id == user_id,
                    AISkillUserOverlay.active == True,  # noqa: E712
                    AISkillUserOverlay.status == "approved",
                )
            ).limit(max_skills)
        )

        index: dict[str, SkillIndexEntry] = {}

        for row in tenant_rows:
            index[row.name] = SkillIndexEntry(
                name=row.name,
                description=row.description,
                scope="tenant",
                overlay_type=row.overlay_type,
            )

        for row in user_rows:
            index[row.name] = SkillIndexEntry(
                name=row.name,
                description=row.description,
                scope="user",
                overlay_type=row.overlay_type,
            )

        return list(index.values())

    # ------------------------------------------------------------------
    # Full context assembly for system prompt injection
    # ------------------------------------------------------------------

    async def assemble_context(
        self, tenant_id: str, user_id: str
    ) -> RuntimeSkillContext:
        """Loads enabled base skills plus all approved active overlays for tenant + user."""
        if not self._settings.skills_enabled:
            return RuntimeSkillContext(
                tenant_id=tenant_id, user_id=user_id,
                tenant_overlays=[], user_overlays=[]
            )

        max_skills = self._settings.skills_max_per_context

        b_result = await self._db.execute(
            select(AISkillBase).where(
                and_(
                    AISkillBase.enabled == True,  # noqa: E712
                    AISkillBase.rule_text != "",
                )
            ).order_by(AISkillBase.id)
        )
        base_skills = [
            BaseSkillInfo.model_validate(row)
            for row in b_result.scalars().all()
        ]

        t_result = await self._db.execute(
            select(AISkillTenantOverlay).where(
                and_(
                    AISkillTenantOverlay.tenant_id == tenant_id,
                    AISkillTenantOverlay.active == True,  # noqa: E712
                    AISkillTenantOverlay.status == "approved",
                )
            ).order_by(AISkillTenantOverlay.created_at).limit(max_skills)
        )
        tenant_overlays = [
            TenantOverlay.model_validate(row)
            for row in t_result.scalars().all()
        ]

        u_result = await self._db.execute(
            select(AISkillUserOverlay).where(
                and_(
                    AISkillUserOverlay.tenant_id == tenant_id,
                    AISkillUserOverlay.user_id == user_id,
                    AISkillUserOverlay.active == True,  # noqa: E712
                    AISkillUserOverlay.status == "approved",
                )
            ).order_by(AISkillUserOverlay.created_at).limit(max_skills)
        )
        user_overlays = [
            UserOverlay.model_validate(row)
            for row in u_result.scalars().all()
        ]

        return RuntimeSkillContext(
            tenant_id=tenant_id,
            user_id=user_id,
            base_skills=base_skills,
            tenant_overlays=tenant_overlays,
            user_overlays=user_overlays,
        )

    # ------------------------------------------------------------------
    # Upsert (create or update by name)
    # ------------------------------------------------------------------

    async def upsert_tenant_overlay(
        self,
        tenant_id: str,
        data: TenantOverlayCreate,
        request_id: Optional[str] = None,
        status: str = "approved",
    ) -> TenantOverlay | None:
        """
        Upserts on (tenant_id, name). Returns None if skills_modification_enabled
        is False, or if a pending upsert would overwrite an approved overlay.
        """
        if not self._settings.skills_modification_enabled:
            logger.info(
                "skill_service: modification disabled, skipping upsert "
                "tenant=%s name=%s", tenant_id, data.name
            )
            return None

        rule_text = data.rule_text[:MAX_RULE_TEXT_CHARS]
        trace_id = _coerce_trace_id(data.source_trace_id, request_id)

        result = await self._db.execute(
            select(AISkillTenantOverlay).where(
                and_(
                    AISkillTenantOverlay.tenant_id == tenant_id,
                    AISkillTenantOverlay.name == data.name,
                )
            )
        )
        existing = result.scalar_one_or_none()

        if existing:
            if status == "pending" and existing.status == "approved":
                logger.info(
                    "skill_service: pending candidate would overwrite approved "
                    "overlay, skipping tenant=%s name=%s", tenant_id, data.name
                )
                return None
            existing.description = data.description
            existing.overlay_type = data.overlay_type
            existing.rule_text = rule_text
            existing.structured_rule = data.structured_rule
            existing.confidence = data.confidence
            if trace_id:
                existing.source_trace_id = trace_id
            existing.status = status
            existing.active = True
            await self._db.commit()
            await self._db.refresh(existing)
            logger.info(
                "skill_service: updated tenant overlay tenant=%s name=%s status=%s",
                tenant_id, data.name, status
            )
            return TenantOverlay.model_validate(existing)

        row = AISkillTenantOverlay(
            tenant_id=tenant_id,
            skill_id=data.skill_id,
            name=data.name,
            description=data.description,
            overlay_type=data.overlay_type,
            rule_text=rule_text,
            structured_rule=data.structured_rule,
            confidence=data.confidence,
            source_trace_id=trace_id,
            created_by=data.created_by,
            status=status,
            active=True,
        )
        self._db.add(row)
        await self._db.commit()
        await self._db.refresh(row)
        logger.info(
            "skill_service: created tenant overlay tenant=%s name=%s",
            tenant_id, data.name
        )
        return TenantOverlay.model_validate(row)

    async def upsert_user_overlay(
        self,
        tenant_id: str,
        user_id: str,
        data: UserOverlayCreate,
        request_id: Optional[str] = None,
    ) -> UserOverlay | None:
        """Returns None if skills_modification_enabled is False. Upserts on (tenant_id, user_id, name)."""
        if not self._settings.skills_modification_enabled:
            return None

        rule_text = data.rule_text[:MAX_RULE_TEXT_CHARS]
        trace_id = _coerce_trace_id(data.source_trace_id, request_id)

        result = await self._db.execute(
            select(AISkillUserOverlay).where(
                and_(
                    AISkillUserOverlay.tenant_id == tenant_id,
                    AISkillUserOverlay.user_id == user_id,
                    AISkillUserOverlay.name == data.name,
                )
            )
        )
        existing = result.scalar_one_or_none()

        if existing:
            existing.description = data.description
            existing.overlay_type = data.overlay_type
            existing.rule_text = rule_text
            existing.structured_rule = data.structured_rule
            existing.confidence = data.confidence
            if trace_id:
                existing.source_trace_id = trace_id
            existing.active = True
            await self._db.commit()
            await self._db.refresh(existing)
            return UserOverlay.model_validate(existing)

        row = AISkillUserOverlay(
            tenant_id=tenant_id,
            user_id=user_id,
            skill_id=data.skill_id,
            name=data.name,
            description=data.description,
            overlay_type=data.overlay_type,
            rule_text=rule_text,
            structured_rule=data.structured_rule,
            confidence=data.confidence,
            source_trace_id=trace_id,
            status="approved",
            active=True,
        )
        self._db.add(row)
        await self._db.commit()
        await self._db.refresh(row)
        return UserOverlay.model_validate(row)

    # ------------------------------------------------------------------
    # Pending approval workflow (auto-captured tenant-scope rules)
    # ------------------------------------------------------------------

    async def list_pending_tenant_overlays(
        self, tenant_id: str
    ) -> list[TenantOverlay]:
        result = await self._db.execute(
            select(AISkillTenantOverlay).where(
                and_(
                    AISkillTenantOverlay.tenant_id == tenant_id,
                    AISkillTenantOverlay.active == True,  # noqa: E712
                    AISkillTenantOverlay.status == "pending",
                )
            ).order_by(AISkillTenantOverlay.created_at)
        )
        return [
            TenantOverlay.model_validate(row)
            for row in result.scalars().all()
        ]

    async def approve_tenant_overlay(
        self, tenant_id: str, name: str
    ) -> TenantOverlay | None:
        """Returns None if modification is disabled or the overlay does not exist."""
        if not self._settings.skills_modification_enabled:
            return None
        result = await self._db.execute(
            select(AISkillTenantOverlay).where(
                and_(
                    AISkillTenantOverlay.tenant_id == tenant_id,
                    AISkillTenantOverlay.name == name,
                )
            )
        )
        row = result.scalar_one_or_none()
        if not row:
            return None
        row.status = "approved"
        row.active = True
        await self._db.commit()
        await self._db.refresh(row)
        logger.info(
            "skill_service: approved tenant overlay tenant=%s name=%s",
            tenant_id, name
        )
        return TenantOverlay.model_validate(row)

    # ------------------------------------------------------------------
    # Deactivate (soft delete)
    # ------------------------------------------------------------------

    async def deactivate_tenant_overlay(
        self, tenant_id: str, name: str
    ) -> bool:
        """Returns True if found and deactivated."""
        if not self._settings.skills_modification_enabled:
            return False
        result = await self._db.execute(
            select(AISkillTenantOverlay).where(
                and_(
                    AISkillTenantOverlay.tenant_id == tenant_id,
                    AISkillTenantOverlay.name == name,
                )
            )
        )
        row = result.scalar_one_or_none()
        if not row:
            return False
        row.active = False
        await self._db.commit()
        return True

    async def deactivate_user_overlay(
        self, tenant_id: str, user_id: str, name: str
    ) -> bool:
        if not self._settings.skills_modification_enabled:
            return False
        result = await self._db.execute(
            select(AISkillUserOverlay).where(
                and_(
                    AISkillUserOverlay.tenant_id == tenant_id,
                    AISkillUserOverlay.user_id == user_id,
                    AISkillUserOverlay.name == name,
                )
            )
        )
        row = result.scalar_one_or_none()
        if not row:
            return False
        row.active = False
        await self._db.commit()
        return True
