from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import get_settings
from app.domain.entity_resolver import EntityResolver
from app.domain.resolver_policy import ResolverPolicy, load_resolver_policy


class DummyCrmClient:
    mode = "coripo_public"

    async def generic_search(self, query: str, scope: str = "all") -> dict[str, Any]:
        if scope == "contacts":
            return {
                "records": [
                    {
                        "id": "11111111-2222-3333-4444-555555555555",
                        "first_name": "Karel",
                        "last_name": "Vybíhal",
                    },
                    {
                        "id": "66666666-2222-3333-4444-555555555555",
                        "first_name": "Jan",
                        "last_name": "Novák",
                    },
                ]
            }
        return {"records": []}

    async def execute_module_action(self, module: str, action: str, data: dict[str, Any]) -> dict[str, Any]:
        return {"records": []}



def test_resolver_policy_is_config_driven(monkeypatch) -> None:
    monkeypatch.setenv("RESOLVER_READ_CONFIDENCE_THRESHOLD", "0.77")
    monkeypatch.setenv("RESOLVER_MUTATION_CONFIDENCE_THRESHOLD", "0.88")
    monkeypatch.setenv("RESOLVER_AMBIGUITY_GAP_THRESHOLD", "0.11")
    get_settings.cache_clear()
    policy = load_resolver_policy()
    assert policy.read_confidence_threshold == 0.77
    assert policy.mutation_confidence_threshold == 0.88
    assert policy.ambiguity_gap_threshold == 0.11
    get_settings.cache_clear()



def test_entity_resolver_threshold_changes_decision_and_logs(caplog) -> None:
    client = DummyCrmClient()

    low_policy = ResolverPolicy(
        read_confidence_threshold=0.60,
        mutation_confidence_threshold=0.60,
        ambiguity_gap_threshold=0.01,
        strict_mutation_confirmation=True,
    )
    strict_policy = ResolverPolicy(
        read_confidence_threshold=0.95,
        mutation_confidence_threshold=0.95,
        ambiguity_gap_threshold=1.10,
        strict_mutation_confirmation=True,
    )

    with caplog.at_level(logging.INFO):
        resolved = asyncio.run(
            EntityResolver(tenant_id="ai-local", crm_client=client, policy=low_policy).resolve_record_by_name(
                scope="contacts",
                name="Karlem Vybíhalem",
                for_mutation=False,
            )
        )
    assert resolved.status == "resolved"
    assert any("Entity resolver decision" in row.message for row in caplog.records)

    ambiguous = asyncio.run(
        EntityResolver(tenant_id="ai-local", crm_client=client, policy=strict_policy).resolve_record_by_name(
            scope="contacts",
            name="Karlem Vybíhalem",
            for_mutation=False,
        )
    )
    assert ambiguous.status == "ambiguous"
