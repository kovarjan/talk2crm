"""
SkillCaptureService — detects user corrections and saves them as skill overlays.

Pipeline:
  1. _has_correction_signal()  — fast regex/keyword check, no LLM
  2. _distill()                — one small LLM call, returns SkillCorrectionCandidate
  3. SkillService.upsert_*()   — saves to DB

Runs as a background task (asyncio.create_task) so it does not block the
response to the user.

IMPORTANT: This service must never raise exceptions that reach the caller.
All errors are caught and logged.
"""
from __future__ import annotations

import json
import logging
import re
import uuid
from typing import Optional

from app.core.config import Settings
from app.core.http import get_shared_http_client
from app.domain.skill_contracts import (
    SkillCorrectionCandidate,
    TenantOverlayCreate,
    UserOverlayCreate,
)
from app.services.skill_service import SkillService

logger = logging.getLogger(__name__)

# Czech and English correction signal patterns
# Conservative to minimise false positives
CORRECTION_PATTERNS = [
    r"\bne[,.]?\s+(to|takhle|takto|tohle)",     # "ne, to je špatně"
    r"\bpříště\b",                                # "příště"
    r"\bvždy\b.{0,30}\b(používej|počítej|dávej|ukazuj|beru|ber)",
    r"\bnikdy\b.{0,30}\b(nepoužívej|nepočítej|nedávej)",
    r"\bpamatuj\s+si\b",
    r"\bzapamatuj\b",
    r"\bsprávně\s+je\b",
    r"\bto\s+dělej\s+jinak\b",
    r"\bchci\s+aby\b",
    r"\bu\s+nás\b.{0,50}\b(je|počítáme|bereme|používáme)",
    r"\bremember\b",
    r"\balways\b.{0,20}\b(use|show|include)",
    r"\bnever\b.{0,20}\b(use|show|include)",
]

_COMPILED = [re.compile(p, re.IGNORECASE) for p in CORRECTION_PATTERNS]

_DISTILL_SYSTEM = """
You are a skill extraction assistant. A user has just corrected an AI assistant's behavior.
Your task: extract a reusable rule from this correction.

Return ONLY a JSON object (no markdown, no explanation) with these fields:
{
  "name": "<short-slug-max-5-words-hyphenated>",
  "description": "<one sentence: when this rule applies>",
  "overlay_type": "<one of: business_rule | field_mapping | entity_alias | workflow_default | response_preference | tool_usage_rule | safety_rule>",
  "rule_text": "<complete instruction for the AI, in the same language as the correction>",
  "structured_rule": null,
  "confidence": <0.0-1.0>,
  "target_skill_id": null
}

overlay_type selection guide:
- business_rule: accounting definitions, what counts as revenue, VAT rules, period definitions
- field_mapping: which CRM field stores which value, field name corrections
- entity_alias: "PANAS means PANAS spol. s r.o.", company nicknames
- workflow_default: default duration, default status, default assignee
- response_preference: how to format/style/length responses
- tool_usage_rule: which tool to use for which task
- safety_rule: what the AI must never do

If the correction is too vague to extract a useful rule, return:
{"confidence": 0.0, "name": "unclear", "description": "", "overlay_type": "response_preference", "rule_text": "", "structured_rule": null, "target_skill_id": null}
"""


class SkillCaptureService:

    def __init__(
        self,
        skill_service: SkillService,
        settings: Settings,
        llm_base_url: str,
        llm_model: str,
        llm_api_key: str = "EMPTY",
    ) -> None:
        self._skill_svc = skill_service
        self._settings = settings
        self._llm_url = llm_base_url.rstrip("/") + "/chat/completions"
        self._llm_model = llm_model
        self._llm_api_key = llm_api_key

    async def maybe_capture(
        self,
        user_message: str,
        agent_response: str,
        tenant_id: str,
        user_id: str,
        request_id: Optional[str] = None,
    ) -> SkillCorrectionCandidate | None:
        """
        Main entry point. Returns the captured candidate or None.
        Never raises — all exceptions are swallowed and logged.
        Call this from a background task after delivering the response.
        """
        if not self._settings.skills_enabled:
            return None
        if not self._settings.skills_capture_enabled:
            return None
        if not self._settings.skills_modification_enabled:
            return None

        try:
            if not self._has_correction_signal(user_message):
                return None

            candidate = await self._distill(user_message, agent_response)
            if candidate is None or candidate.confidence < self._settings.skills_min_confidence:
                logger.debug(
                    "skill_capture: low confidence or None for tenant=%s user=%s",
                    tenant_id, user_id
                )
                return None

            await self._save(candidate, tenant_id, user_id, request_id)
            logger.info(
                "skill_capture: saved %s scope=%s tenant=%s user=%s confidence=%.2f",
                candidate.name, candidate.recommended_scope,
                tenant_id, user_id, candidate.confidence
            )
            return candidate

        except Exception:
            logger.exception(
                "skill_capture: unhandled error tenant=%s user=%s", tenant_id, user_id
            )
            return None

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _has_correction_signal(self, text: str) -> bool:
        return any(p.search(text) for p in _COMPILED)

    async def _distill(
        self, user_message: str, agent_response: str
    ) -> SkillCorrectionCandidate | None:
        """One small LLM call. max_tokens=400 keeps this fast and cheap."""
        prompt = (
            f"User message:\n{user_message}\n\n"
            f"Previous AI response:\n{agent_response[:500]}"
        )

        try:
            client = get_shared_http_client()
            resp = await client.post(
                self._llm_url,
                headers={"Authorization": f"Bearer {self._llm_api_key}"},
                timeout=15.0,
                json={
                    "model": self._llm_model,
                    "max_tokens": 400,
                    "temperature": 0.1,
                    "messages": [
                        {"role": "system", "content": _DISTILL_SYSTEM},
                        {"role": "user", "content": prompt},
                    ],
                },
            )
            resp.raise_for_status()
            data = resp.json()
            raw = data["choices"][0]["message"]["content"].strip()

            # Strip markdown fences if model wraps output
            raw = re.sub(r"^```json\s*|```$", "", raw, flags=re.MULTILINE).strip()

            parsed = json.loads(raw)
            return SkillCorrectionCandidate(**parsed)

        except Exception as exc:
            logger.warning("skill_capture: distill failed: %s", exc)
            return None

    @staticmethod
    def _parse_uuid(value: Optional[str]) -> Optional[uuid.UUID]:
        if not value:
            return None
        try:
            return uuid.UUID(str(value))
        except (ValueError, AttributeError):
            return None

    async def _save(
        self,
        candidate: SkillCorrectionCandidate,
        tenant_id: str,
        user_id: str,
        request_id: Optional[str],
    ) -> None:
        scope = candidate.recommended_scope
        trace_id = self._parse_uuid(request_id)

        if scope == "tenant":
            # Tenant-scope rules affect every user of the tenant — auto-captured
            # candidates stay pending until approved via the API.
            await self._skill_svc.upsert_tenant_overlay(
                tenant_id=tenant_id,
                status="pending",
                data=TenantOverlayCreate(
                    name=candidate.name,
                    description=candidate.description,
                    overlay_type=candidate.overlay_type,
                    rule_text=candidate.rule_text,
                    structured_rule=candidate.structured_rule,
                    confidence=candidate.confidence,
                    skill_id=candidate.target_skill_id,
                    created_by=user_id,
                    source_trace_id=trace_id,
                ),
                request_id=request_id,
            )
        else:
            await self._skill_svc.upsert_user_overlay(
                tenant_id=tenant_id,
                user_id=user_id,
                data=UserOverlayCreate(
                    name=candidate.name,
                    description=candidate.description,
                    overlay_type=candidate.overlay_type,
                    rule_text=candidate.rule_text,
                    structured_rule=candidate.structured_rule,
                    confidence=candidate.confidence,
                    skill_id=candidate.target_skill_id,
                    source_trace_id=trace_id,
                ),
                request_id=request_id,
            )
