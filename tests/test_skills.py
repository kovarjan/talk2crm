"""
Tests for the multi-tenant skills system.
Self-contained — uses mocks, no live network or database.
"""
from __future__ import annotations

import uuid
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.domain.skill_contracts import (
    SCOPE_BY_OVERLAY_TYPE,
    BaseSkillInfo,
    RuntimeSkillContext,
    SkillCorrectionCandidate,
    TenantOverlay,
    TenantOverlayCreate,
    UserOverlay,
    UserOverlayCreate,
)
from app.engine.agent import _fallback_base_skills
from app.engine.base_skills import BASE_SKILLS
from app.services.skill_capture_service import SkillCaptureService, _COMPILED
from app.services.skill_service import SkillService


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tenant_overlay(**kwargs) -> TenantOverlay:
    defaults = dict(
        id=uuid.uuid4(),
        tenant_id="t1",
        skill_id=None,
        name="test-overlay",
        description="desc",
        overlay_type="business_rule",
        rule_text="rule text",
        structured_rule=None,
        confidence=None,
        source_trace_id=None,
        created_by=None,
        active=True,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )
    defaults.update(kwargs)
    return TenantOverlay(**defaults)


def _user_overlay(**kwargs) -> UserOverlay:
    defaults = dict(
        id=uuid.uuid4(),
        tenant_id="t1",
        user_id="u1",
        skill_id=None,
        name="test-user-overlay",
        description="desc",
        overlay_type="response_preference",
        rule_text="user rule",
        structured_rule=None,
        confidence=None,
        source_trace_id=None,
        active=True,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )
    defaults.update(kwargs)
    return UserOverlay(**defaults)


def _make_capture_svc(skill_svc=None, settings=None):
    if settings is None:
        settings = MagicMock()
        settings.skills_enabled = True
        settings.skills_capture_enabled = True
        settings.skills_modification_enabled = True
        settings.skills_min_confidence = 0.5
    if skill_svc is None:
        skill_svc = MagicMock()
        skill_svc.upsert_tenant_overlay = AsyncMock(return_value=None)
        skill_svc.upsert_user_overlay = AsyncMock(return_value=None)
    return SkillCaptureService(
        skill_service=skill_svc,
        settings=settings,
        llm_base_url="http://localhost:11434/v1",
        llm_model="qwen3",
    )


# ---------------------------------------------------------------------------
# Contracts — scope derivation
# ---------------------------------------------------------------------------

class TestSkillContracts:
    def test_scope_derivation_business_rule(self):
        c = SkillCorrectionCandidate(
            name="test", description="desc", overlay_type="business_rule",
            rule_text="rule", confidence=0.9
        )
        assert c.recommended_scope == "tenant"

    def test_scope_derivation_response_preference(self):
        c = SkillCorrectionCandidate(
            name="test", description="desc", overlay_type="response_preference",
            rule_text="rule", confidence=0.9
        )
        assert c.recommended_scope == "user"

    def test_scope_derivation_field_mapping(self):
        assert SCOPE_BY_OVERLAY_TYPE["field_mapping"] == "tenant"

    def test_scope_derivation_workflow_default(self):
        assert SCOPE_BY_OVERLAY_TYPE["workflow_default"] == "user"

    def test_scope_derivation_entity_alias(self):
        assert SCOPE_BY_OVERLAY_TYPE["entity_alias"] == "tenant"

    def test_scope_derivation_safety_rule(self):
        assert SCOPE_BY_OVERLAY_TYPE["safety_rule"] == "tenant"

    def test_scope_derivation_tool_usage_rule(self):
        assert SCOPE_BY_OVERLAY_TYPE["tool_usage_rule"] == "tenant"

    def test_all_overlay_types_have_scope(self):
        for ot in [
            "business_rule", "field_mapping", "entity_alias",
            "workflow_default", "response_preference",
            "tool_usage_rule", "safety_rule",
        ]:
            assert ot in SCOPE_BY_OVERLAY_TYPE, f"{ot} missing from SCOPE_BY_OVERLAY_TYPE"

    def test_runtime_context_to_prompt_block_empty(self):
        ctx = RuntimeSkillContext(
            tenant_id="t1", user_id="u1",
            tenant_overlays=[], user_overlays=[]
        )
        assert ctx.to_prompt_block() == ""

    def test_runtime_context_prompt_block_with_tenant_overlay(self):
        o = _tenant_overlay(name="invoice-rule", rule_text="Vždy počítej bez DPH.")
        ctx = RuntimeSkillContext(
            tenant_id="t1", user_id="u1",
            tenant_overlays=[o], user_overlays=[]
        )
        block = ctx.to_prompt_block()
        assert "invoice-rule" in block
        assert "Vždy počítej bez DPH." in block
        assert "Pravidla organizace" in block
        assert "Pravidla a dovednosti" in block

    def test_runtime_context_prompt_block_with_user_overlay(self):
        o = _user_overlay(name="kratke-odpovedi", rule_text="Odpovídej stručně.")
        ctx = RuntimeSkillContext(
            tenant_id="t1", user_id="u1",
            tenant_overlays=[], user_overlays=[o]
        )
        block = ctx.to_prompt_block()
        assert "kratke-odpovedi" in block
        assert "Odpovídej stručně." in block
        assert "osobní preference" in block

    def test_runtime_context_prompt_block_both(self):
        t = _tenant_overlay(name="tenant-rule", rule_text="tenant rule text")
        u = _user_overlay(name="user-pref", rule_text="user pref text")
        ctx = RuntimeSkillContext(
            tenant_id="t1", user_id="u1",
            tenant_overlays=[t], user_overlays=[u]
        )
        block = ctx.to_prompt_block()
        assert "tenant-rule" in block
        assert "user-pref" in block
        assert "Pravidla organizace mají přednost" in block

    def test_runtime_context_prompt_block_with_base_skills(self):
        base = BaseSkillInfo(
            id="contact-selection-rules",
            name="Výběr kontaktu/firmy",
            description="desc",
            rule_text="- Pokud najdeš více IDENTICKÝCH kontaktů, vyber první.",
        )
        ctx = RuntimeSkillContext(
            tenant_id="t1", user_id="u1",
            base_skills=[base], tenant_overlays=[], user_overlays=[]
        )
        block = ctx.to_prompt_block()
        assert "Základní pravidla" in block
        assert "Výběr kontaktu/firmy" in block
        assert "IDENTICKÝCH" in block


class TestBaseSkillDefinitions:
    def test_base_skills_have_required_fields(self):
        for skill in BASE_SKILLS:
            assert skill["id"]
            assert skill["name"]
            assert skill["description"]
            assert skill["rule_text"]

    def test_fallback_base_skills_render(self):
        ctx = RuntimeSkillContext(
            tenant_id="t1", user_id="u1",
            base_skills=_fallback_base_skills(),
            tenant_overlays=[], user_overlays=[],
        )
        block = ctx.to_prompt_block()
        # Migrated hardcoded prompt rules must be present via base skills
        assert "IDENTICKÝCH" in block          # contact selection rules
        assert "my_meetings_tool" in block      # meeting query rules
        assert "amount_without_vat" in block    # module/field guidance


# ---------------------------------------------------------------------------
# Correction signal detection
# ---------------------------------------------------------------------------

class TestCorrectionSignalDetection:
    def _detect(self, text: str) -> bool:
        return any(p.search(text) for p in _COMPILED)

    def test_detects_priiste(self):
        assert self._detect("příště mi výsledky dávej stručně") is True

    def test_detects_pamatuj_si(self):
        assert self._detect("Pamatuj si že fakturaci počítáme bez DPH") is True

    def test_detects_u_nas(self):
        assert self._detect("u nás fakturaci počítáme bez zálohovek") is True

    def test_detects_vzdy(self):
        assert self._detect("vždy používej pole amount_without_vat") is True

    def test_detects_nikdy(self):
        assert self._detect("nikdy nepočítej zálohy do fakturace") is True

    def test_detects_spravne_je(self):
        assert self._detect("správně je amount_without_vat ne amount_total") is True

    def test_detects_english_remember(self):
        assert self._detect("remember that invoices use amount_without_vat") is True

    def test_detects_english_always(self):
        assert self._detect("always include VAT in totals") is True

    def test_detects_english_never(self):
        assert self._detect("never show cancelled invoices") is True

    def test_detects_zapamatuj(self):
        assert self._detect("zapamatuj si tento postup") is True

    def test_no_signal_normal_query(self):
        assert self._detect("kolik jsme vyfakturovali letos") is False

    def test_no_signal_question(self):
        assert self._detect("jaká je průměrná hodnota nabídek?") is False

    def test_no_signal_crm_query(self):
        assert self._detect("zobraz mi schůzky za tento týden") is False


# ---------------------------------------------------------------------------
# SkillCaptureService
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_capture_skipped_when_skills_disabled():
    settings = MagicMock()
    settings.skills_enabled = False
    svc = _make_capture_svc(settings=settings)
    result = await svc.maybe_capture("příště stručně", "odpověď", "t1", "u1")
    assert result is None


@pytest.mark.asyncio
async def test_capture_skipped_when_capture_disabled():
    settings = MagicMock()
    settings.skills_enabled = True
    settings.skills_capture_enabled = False
    svc = _make_capture_svc(settings=settings)
    result = await svc.maybe_capture("příště stručně", "odpověď", "t1", "u1")
    assert result is None


@pytest.mark.asyncio
async def test_capture_skipped_when_modification_disabled():
    settings = MagicMock()
    settings.skills_enabled = True
    settings.skills_capture_enabled = True
    settings.skills_modification_enabled = False
    svc = _make_capture_svc(settings=settings)
    result = await svc.maybe_capture("příště stručně", "odpověď", "t1", "u1")
    assert result is None


@pytest.mark.asyncio
async def test_capture_skipped_when_no_signal():
    svc = _make_capture_svc()
    result = await svc.maybe_capture("kolik jsme vyfakturovali letos?", "odpověď", "t1", "u1")
    assert result is None


@pytest.mark.asyncio
async def test_capture_saves_tenant_scope_for_business_rule():
    skill_svc = MagicMock()
    skill_svc.upsert_tenant_overlay = AsyncMock(return_value=MagicMock())
    skill_svc.upsert_user_overlay = AsyncMock(return_value=None)

    candidate = SkillCorrectionCandidate(
        name="invoice-bez-dph",
        description="Fakturaci počítat bez DPH",
        overlay_type="business_rule",
        rule_text="Vždy počítej fakturaci bez DPH.",
        confidence=0.9,
    )

    svc = _make_capture_svc(skill_svc=skill_svc)

    with patch.object(svc, "_distill", AsyncMock(return_value=candidate)):
        result = await svc.maybe_capture(
            "u nás fakturaci počítáme bez DPH",
            "Celková fakturace je 100 000 Kč.",
            "tenant1", "user1",
        )

    assert result is not None
    skill_svc.upsert_tenant_overlay.assert_called_once()
    skill_svc.upsert_user_overlay.assert_not_called()


@pytest.mark.asyncio
async def test_capture_saves_user_scope_for_preference():
    skill_svc = MagicMock()
    skill_svc.upsert_tenant_overlay = AsyncMock(return_value=None)
    skill_svc.upsert_user_overlay = AsyncMock(return_value=MagicMock())

    candidate = SkillCorrectionCandidate(
        name="kratsi-odpovedi",
        description="Krátké odpovědi",
        overlay_type="response_preference",
        rule_text="Odpovídej stručně.",
        confidence=0.85,
    )

    svc = _make_capture_svc(skill_svc=skill_svc)

    with patch.object(svc, "_distill", AsyncMock(return_value=candidate)):
        result = await svc.maybe_capture(
            "příště mi odpovídej stručně",
            "Tady je dlouhá odpověď...",
            "tenant1", "user1",
        )

    assert result is not None
    skill_svc.upsert_user_overlay.assert_called_once()
    skill_svc.upsert_tenant_overlay.assert_not_called()


@pytest.mark.asyncio
async def test_capture_skips_low_confidence():
    svc = _make_capture_svc()
    low_conf = SkillCorrectionCandidate(
        name="unclear", description="", overlay_type="response_preference",
        rule_text="", confidence=0.3
    )
    with patch.object(svc, "_distill", AsyncMock(return_value=low_conf)):
        result = await svc.maybe_capture("příště jinak", "odpověď", "t1", "u1")
    assert result is None


@pytest.mark.asyncio
async def test_capture_skips_none_from_distill():
    svc = _make_capture_svc()
    with patch.object(svc, "_distill", AsyncMock(return_value=None)):
        result = await svc.maybe_capture("příště jinak", "odpověď", "t1", "u1")
    assert result is None


@pytest.mark.asyncio
async def test_capture_never_raises():
    """SkillCaptureService must swallow all exceptions."""
    svc = _make_capture_svc()
    with patch.object(svc, "_distill", AsyncMock(side_effect=RuntimeError("boom"))):
        result = await svc.maybe_capture("příště jinak", "odpověď", "t1", "u1")
    assert result is None


@pytest.mark.asyncio
async def test_capture_passes_request_id():
    skill_svc = MagicMock()
    skill_svc.upsert_tenant_overlay = AsyncMock(return_value=MagicMock())
    skill_svc.upsert_user_overlay = AsyncMock(return_value=None)

    candidate = SkillCorrectionCandidate(
        name="test", description="test", overlay_type="business_rule",
        rule_text="rule", confidence=0.9
    )
    svc = _make_capture_svc(skill_svc=skill_svc)

    req_id = str(uuid.uuid4())
    with patch.object(svc, "_distill", AsyncMock(return_value=candidate)):
        await svc.maybe_capture(
            "u nás vždy počítáme bez DPH", "ok", "t1", "u1",
            request_id=req_id,
        )

    call_kwargs = skill_svc.upsert_tenant_overlay.call_args
    assert call_kwargs.kwargs.get("request_id") == req_id


@pytest.mark.asyncio
async def test_capture_tenant_scope_is_pending():
    """Auto-captured tenant-scope rules must not be auto-approved."""
    skill_svc = MagicMock()
    skill_svc.upsert_tenant_overlay = AsyncMock(return_value=MagicMock())
    skill_svc.upsert_user_overlay = AsyncMock(return_value=None)

    candidate = SkillCorrectionCandidate(
        name="invoice-bez-dph",
        description="Fakturaci počítat bez DPH",
        overlay_type="business_rule",
        rule_text="Vždy počítej fakturaci bez DPH.",
        confidence=0.9,
    )
    svc = _make_capture_svc(skill_svc=skill_svc)

    with patch.object(svc, "_distill", AsyncMock(return_value=candidate)):
        await svc.maybe_capture(
            "u nás fakturaci počítáme bez DPH", "odpověď", "t1", "u1"
        )

    assert skill_svc.upsert_tenant_overlay.call_args.kwargs.get("status") == "pending"


# ---------------------------------------------------------------------------
# SkillService — pending upsert must not clobber approved overlays
# ---------------------------------------------------------------------------

def _make_skill_service(existing_row=None, settings=None) -> SkillService:
    if settings is None:
        settings = MagicMock()
        settings.skills_enabled = True
        settings.skills_modification_enabled = True
        settings.skills_max_per_context = 20
    db = MagicMock()
    result = MagicMock()
    result.scalar_one_or_none = MagicMock(return_value=existing_row)
    db.execute = AsyncMock(return_value=result)
    db.commit = AsyncMock()
    db.refresh = AsyncMock()
    db.add = MagicMock()
    return SkillService(db, settings)


@pytest.mark.asyncio
async def test_pending_upsert_skips_approved_overlay():
    existing = MagicMock()
    existing.status = "approved"
    svc = _make_skill_service(existing_row=existing)

    result = await svc.upsert_tenant_overlay(
        "t1",
        TenantOverlayCreate(
            name="rule", description="d", overlay_type="business_rule",
            rule_text="new text",
        ),
        status="pending",
    )
    assert result is None
    svc._db.commit.assert_not_called()


@pytest.mark.asyncio
async def test_pending_upsert_updates_pending_overlay():
    existing = MagicMock()
    existing.status = "pending"
    svc = _make_skill_service(existing_row=existing)

    with patch(
        "app.domain.skill_contracts.TenantOverlay.model_validate",
        MagicMock(return_value=MagicMock()),
    ):
        result = await svc.upsert_tenant_overlay(
            "t1",
            TenantOverlayCreate(
                name="rule", description="d", overlay_type="business_rule",
                rule_text="new text",
            ),
            status="pending",
        )
    assert result is not None
    assert existing.status == "pending"
    svc._db.commit.assert_called_once()


@pytest.mark.asyncio
async def test_upsert_stores_uuid_trace_id_not_string():
    """source_trace_id column is UUID — a raw request string must be parsed."""
    existing = MagicMock()
    existing.status = "approved"
    svc = _make_skill_service(existing_row=existing)

    req_id = str(uuid.uuid4())
    with patch(
        "app.domain.skill_contracts.TenantOverlay.model_validate",
        MagicMock(return_value=MagicMock()),
    ):
        await svc.upsert_tenant_overlay(
            "t1",
            TenantOverlayCreate(
                name="rule", description="d", overlay_type="business_rule",
                rule_text="text",
            ),
            request_id=req_id,
        )
    assert isinstance(existing.source_trace_id, uuid.UUID)
    assert str(existing.source_trace_id) == req_id


# ---------------------------------------------------------------------------
# Structured chat history for the agent
# ---------------------------------------------------------------------------

class TestHistoryToMessages:
    def test_alternating_turns(self):
        from langchain_core.messages import AIMessage, HumanMessage
        from app.engine.agent import _history_to_messages

        history = [
            {"role": "user", "content": "můj tajný kód je Ax554Cpm43"},
            {"role": "assistant", "content": "Rozumím."},
            {"role": "user", "content": "vytvoř poznámku"},
        ]
        msgs = _history_to_messages(history)
        assert len(msgs) == 3
        assert isinstance(msgs[0], HumanMessage)
        assert isinstance(msgs[1], AIMessage)
        assert isinstance(msgs[2], HumanMessage)
        assert "Ax554Cpm43" in msgs[0].content

    def test_empty_history(self):
        from app.engine.agent import _history_to_messages
        assert _history_to_messages(None) == []
        assert _history_to_messages([]) == []

    def test_truncates_long_content_and_skips_empty(self):
        from app.engine.agent import _history_to_messages
        history = [
            {"role": "user", "content": "x" * 5000},
            {"role": "assistant", "content": ""},
        ]
        msgs = _history_to_messages(history)
        assert len(msgs) == 1
        assert len(msgs[0].content) < 1600
