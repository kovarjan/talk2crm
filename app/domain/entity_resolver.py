from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any

from app.core.logging import get_logger
from app.domain.contracts import EntityResolution
from app.domain.resolver_policy import ResolverPolicy, load_resolver_policy
from app.services.crm_client import SugarClient


logger = get_logger(__name__)
_CRM_ID_RE = re.compile(
    r"^(?:[0-9a-fA-F]{32}|[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12})$"
)
_PLACEHOLDER_ID_RE = re.compile(r"^[A-Z_]+_ID$")


@dataclass
class EntityResolver:
    tenant_id: str
    crm_client: SugarClient
    policy: ResolverPolicy

    @classmethod
    def from_settings(cls, *, tenant_id: str, crm_client: SugarClient) -> "EntityResolver":
        return cls(tenant_id=tenant_id, crm_client=crm_client, policy=load_resolver_policy())

    @staticmethod
    def _normalize_text(value: str) -> str:
        lowered = (value or "").strip().lower()
        unaccented = "".join(
            c for c in unicodedata.normalize("NFD", lowered) if unicodedata.category(c) != "Mn"
        )
        return re.sub(r"[^a-z0-9]+", " ", unaccented).strip()

    @staticmethod
    def _is_valid_crm_id(value: str | None) -> bool:
        text = str(value or "").strip()
        if not text:
            return False
        if _PLACEHOLDER_ID_RE.match(text):
            return False
        return bool(_CRM_ID_RE.match(text))

    @classmethod
    def expand_name_variants(cls, name: str) -> list[str]:
        base = str(name or "").strip()
        if not base:
            return []

        def token_variants(token: str) -> list[str]:
            raw = token.strip()
            if not raw:
                return []
            out = [raw]
            norm = cls._normalize_text(raw)
            if norm.endswith("ovou") and len(raw) > 4:
                out.append(raw[:-4] + "ova")
            elif norm.endswith("ou") and len(raw) > 2:
                out.append(raw[:-2] + "a")
            if norm.endswith("lem") and len(raw) > 4:
                out.append(raw[:-3] + "el")
            if norm.endswith("kem") and len(raw) > 4:
                out.append(raw[:-3] + "ek")
            if norm.endswith("rem") and len(raw) > 4:
                out.append(raw[:-3] + "r")
            if norm.endswith("em") and len(raw) > 4:
                out.append(raw[:-2])
            return list(dict.fromkeys(item for item in out if item.strip()))

        variants = [base]
        tokens = [token for token in re.split(r"\s+", base) if token]
        if tokens:
            combos: list[list[str]] = [[]]
            for token in tokens:
                options = token_variants(token)
                next_combos: list[list[str]] = []
                for prefix in combos:
                    for option in options:
                        next_combos.append([*prefix, option])
                combos = next_combos[:16]
            for combo in combos:
                variant = " ".join(combo).strip()
                if variant:
                    variants.append(variant)

        return list(dict.fromkeys(v for v in variants if v.strip()))

    @classmethod
    def _extract_records(cls, payload: Any) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []

        def collect(value: Any) -> None:
            if isinstance(value, dict):
                direct_records = value.get("records")
                if isinstance(direct_records, list):
                    for item in direct_records:
                        if isinstance(item, dict):
                            records.append(item)
                for key in ("result", "data", "message", "contacts", "accounts", "meetings", "crm"):
                    nested = value.get(key)
                    if nested is not None:
                        collect(nested)
                return
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        records.append(item)

        collect(payload)
        unique: dict[str, dict[str, Any]] = {}
        for row in records:
            row_id = str(row.get("id") or row.get("record_id") or "").strip()
            if not cls._is_valid_crm_id(row_id):
                continue
            if row_id not in unique:
                unique[row_id] = row
        return list(unique.values())

    @staticmethod
    def _record_label(record: dict[str, Any]) -> str:
        if record.get("name"):
            return str(record["name"])
        first = str(record.get("first_name") or "").strip()
        last = str(record.get("last_name") or "").strip()
        if first or last:
            return f"{first} {last}".strip()
        account_name = str(record.get("account_name") or record.get("company") or "").strip()
        if account_name:
            return account_name
        return ""

    @classmethod
    def _score_match(cls, search_name: str, record: dict[str, Any]) -> int:
        candidate = cls._record_label(record)
        if not candidate:
            return 0
        n_search = cls._normalize_text(search_name)
        n_candidate = cls._normalize_text(candidate)
        if not n_search or not n_candidate:
            return 0
        if n_search == n_candidate:
            return 100
        if n_search in n_candidate or n_candidate in n_search:
            return 80
        s_tokens = set(n_search.split())
        c_tokens = set(n_candidate.split())
        if not s_tokens:
            return 0
        overlap = len(s_tokens & c_tokens) / len(s_tokens)
        score = int(overlap * 60)

        seq_ratio = SequenceMatcher(None, n_search, n_candidate).ratio()
        if seq_ratio >= 0.72:
            score = max(score, int(seq_ratio * 85))
        return score

    @staticmethod
    def _lookup_filter_for_scope(scope: str, query: str) -> dict[str, Any]:
        value = str(query or "").strip()
        if not value:
            return {"operator": "and", "operands": []}

        if scope == "contacts":
            return {
                "operator": "and",
                "operands": [
                    {
                        "operator": "or",
                        "operands": [
                            {"field": "name", "type": "cont", "value": value},
                            {"field": "first_name", "type": "cont", "value": value},
                            {"field": "last_name", "type": "cont", "value": value},
                        ],
                    }
                ],
            }
        return {
            "operator": "and",
            "operands": [
                {
                    "operator": "or",
                    "operands": [
                        {"field": "name", "type": "cont", "value": value},
                    ],
                }
            ],
        }

    @staticmethod
    def _fallback_response_fields(scope: str) -> list[str]:
        normalized_scope = str(scope or "").strip().lower()
        if normalized_scope == "contacts":
            return [
                "id",
                "name",
                "first_name",
                "last_name",
                "account_name",
                "company",
                "email1",
                "phone_mobile",
            ]
        if normalized_scope == "accounts":
            return [
                "id",
                "name",
                "account_name",
                "billing_address_city",
            ]
        return ["id", "name"]

    async def resolve_record_by_name(
        self,
        *,
        scope: str,
        name: str,
        for_mutation: bool = False,
    ) -> EntityResolution:
        raw_query = str(name or "").strip()
        normalized_query = self._normalize_text(raw_query)
        variants = self.expand_name_variants(raw_query)
        module_by_scope = {"contacts": "Contacts", "accounts": "Accounts"}
        module_name = module_by_scope.get(scope.strip().lower())
        stage_candidate_counts: dict[str, int] = {
            "generic": 0,
            "fallback_list": 0,
            "dropped_nameless": 0,
            "dropped_zero_score": 0,
            "kept_scored": 0,
        }

        all_scored: list[tuple[int, dict[str, Any], str]] = []
        for candidate_name in variants:
            try:
                payload = await self.crm_client.generic_search(query=candidate_name, scope=scope)
                records = self._extract_records(payload)
                stage_candidate_counts["generic"] += len(records)
            except Exception:
                logger.exception(
                    "Entity resolve %s generic lookup failed tenant=%s query=%s",
                    scope,
                    self.tenant_id,
                    candidate_name,
                )
                records = []

            if not records and module_name:
                try:
                    fallback_payload = await self.crm_client.execute_module_action(
                        module=module_name,
                        action="list",
                        data={
                            "query": candidate_name,
                            "q": candidate_name,
                            "max_results": 50,
                            "savedSearch": False,
                            "filter": self._lookup_filter_for_scope(scope=scope, query=candidate_name),
                            "include_field_names": False,
                            "response_fields": self._fallback_response_fields(scope),
                        },
                    )
                    records = self._extract_records(fallback_payload)
                    stage_candidate_counts["fallback_list"] += len(records)
                except Exception:
                    logger.exception(
                        "Entity resolve %s fallback lookup failed tenant=%s query=%s",
                        scope,
                        self.tenant_id,
                        candidate_name,
                    )

            for row in records:
                if not self._record_label(row):
                    stage_candidate_counts["dropped_nameless"] += 1
                    continue
                score = self._score_match(candidate_name, row)
                if score <= 0:
                    stage_candidate_counts["dropped_zero_score"] += 1
                    continue
                stage_candidate_counts["kept_scored"] += 1
                all_scored.append((score, row, candidate_name))

        if not all_scored:
            reason = "no_scored_candidates" if (stage_candidate_counts["generic"] or stage_candidate_counts["fallback_list"]) else "no_candidates"
            logger.info(
                "Entity resolver decision tenant=%s scope=%s raw_query=%s normalized_query=%s variants=%s stage_counts=%s candidates=%s reason=%s",
                self.tenant_id,
                scope,
                raw_query,
                normalized_query,
                variants[:8],
                stage_candidate_counts,
                0,
                reason,
            )
            return EntityResolution(
                status="unresolved",
                selected=None,
                candidates=[],
                confidence=0.0,
                reason=reason,
            )

        # Deduplicate across name variants by record id and keep the strongest score.
        deduped: dict[str, tuple[int, dict[str, Any], str]] = {}
        for score, row, query_text in all_scored:
            row_id = str(row.get("id") or "").strip()
            if not row_id:
                continue
            prev = deduped.get(row_id)
            if prev is None or score > prev[0]:
                deduped[row_id] = (score, row, query_text)

        ranked = sorted(deduped.values(), key=lambda item: item[0], reverse=True)
        if not ranked:
            return EntityResolution(
                status="unresolved",
                selected=None,
                candidates=[],
                confidence=0.0,
                reason="no_ranked_candidates",
            )

        top_score, top_row, top_query = ranked[0]
        second_score = ranked[1][0] if len(ranked) > 1 else 0

        confidence = max(0.0, min(1.0, float(top_score) / 100.0))
        threshold = (
            self.policy.mutation_confidence_threshold
            if for_mutation
            else self.policy.read_confidence_threshold
        )
        gap = confidence - max(0.0, min(1.0, float(second_score) / 100.0))
        candidate_rows = [row for _, row, _ in ranked[:5]]
        top_candidates = [
            {
                "id": str(row.get("id") or "").strip(),
                "name": self._record_label(row),
                "query_variant": query_variant,
                "score": score,
            }
            for score, row, query_variant in ranked[:5]
        ]

        decision_reason = "scored"
        decision_status = "resolved"
        if confidence < threshold or gap < self.policy.ambiguity_gap_threshold:
            decision_reason = "below_threshold_or_gap"
            decision_status = "ambiguous"
            result = EntityResolution(
                status="ambiguous",
                selected=None,
                candidates=candidate_rows,
                confidence=confidence,
                reason="below_threshold_or_gap",
            )
            logger.info(
                "Entity resolver decision tenant=%s scope=%s raw_query=%s normalized_query=%s top_variant=%s top_score=%s second_score=%s confidence=%.3f threshold=%.3f gap=%.3f stage_counts=%s variants=%s top_candidates=%s status=%s reason=%s",
                self.tenant_id,
                scope,
                raw_query,
                normalized_query,
                top_query,
                top_score,
                second_score,
                confidence,
                threshold,
                gap,
                stage_candidate_counts,
                variants[:8],
                top_candidates,
                decision_status,
                decision_reason,
            )
            return result

        result = EntityResolution(
            status="resolved",
            selected=top_row,
            candidates=candidate_rows,
            confidence=confidence,
            reason="scored",
        )
        logger.info(
            "Entity resolver decision tenant=%s scope=%s raw_query=%s normalized_query=%s top_variant=%s top_score=%s second_score=%s confidence=%.3f threshold=%.3f gap=%.3f stage_counts=%s variants=%s top_candidates=%s status=%s reason=%s",
            self.tenant_id,
            scope,
            raw_query,
            normalized_query,
            top_query,
            top_score,
            second_score,
            confidence,
            threshold,
            gap,
            stage_candidate_counts,
            variants[:8],
            top_candidates,
            decision_status,
            decision_reason,
        )
        return result
