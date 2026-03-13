from __future__ import annotations

import json
import re
import unicodedata
from difflib import SequenceMatcher
from datetime import datetime, timedelta
from typing import Any

from langchain_core.tools import tool

from app.core.config import get_settings
from app.engine.rag import TenantRAGService
from app.services.crm_read_service import CRMReadHelpers, CRMReadService
from app.services.crm_client import SugarClient
from app.services.crm_write_service import CRMWriteService
from app.engine.filter_builder import FilterSpec, build_filter, build_order
from app.engine.tool_logger import ToolCallLogger

_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)


def _normalize_text(value: str) -> str:
    lowered = (value or "").strip().lower()
    unaccented = "".join(
        c for c in unicodedata.normalize("NFD", lowered) if unicodedata.category(c) != "Mn"
    )
    return re.sub(r"[^a-z0-9]+", " ", unaccented).strip()


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


_COMPANY_PREFIX_RE = re.compile(
    r"^\s*(?:(?:z|ze|u|k|ke|v|ve)\s+)?(?:firma|firmy|firmu|firmě|firme)\s+",
    re.IGNORECASE,
)

# CRM module - filtering throws out "firma Cemat" makes it only "Cemat"
def _normalize_account_lookup_query(query: str) -> str:
    raw = _safe_text(query)
    if not raw:
        return ""
    cleaned = _COMPANY_PREFIX_RE.sub("", raw, count=1).strip(" \t,;:-")
    return cleaned or raw


def _infer_module_from_intent(input_text: str, requested_module: str) -> str:
    normalized = _normalize_text(input_text)
    module_norm = (requested_module or "").strip().lower()
    module_aliases = {
        "meeting": "Meetings",
        "meetings": "Meetings",
        "schuzka": "Meetings",
        "schuzky": "Meetings",
        "call": "Calls",
        "calls": "Calls",
        "hovor": "Calls",
        "task": "Tasks",
        "tasks": "Tasks",
        "ukol": "Tasks",
        "note": "Notes",
        "notes": "Notes",
        "poznamka": "Notes",
        "poznamky": "Notes",
    }

    call_tokens = ("hovor", "telefonat", "zavolej", "volat", "call")
    task_tokens = ("ukol", "task", "todo", "pripomen", "follow up")
    note_tokens = ("poznamk", "zapis", "zapisek", "note")
    meeting_tokens = ("schuzk", "meeting")

    has_call = any(token in normalized for token in call_tokens)
    has_task = any(token in normalized for token in task_tokens)
    has_note = any(token in normalized for token in note_tokens)
    has_meeting = any(token in normalized for token in meeting_tokens)

    explicit_module = module_aliases.get(module_norm)
    if explicit_module:
        # Resolve explicit LLM/user module conflicts by trusting clear user intent tokens.
        if explicit_module == "Meetings" and has_call and not has_meeting:
            return "Calls"
        if explicit_module == "Calls" and has_meeting and not has_call:
            return "Meetings"
        return explicit_module
    if module_norm:
        return requested_module

    if has_task:
        return "Tasks"
    if has_call:
        return "Calls"
    if has_meeting:
        return "Meetings"
    if has_note:
        return "Notes"

    return "Meetings"


def _parse_datetime(value: Any) -> datetime | None:
    raw = _safe_text(value)
    if not raw:
        return None
    for fmt in (
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d",
    ):
        try:
            return datetime.strptime(raw, fmt)
        except ValueError:
            continue
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        return parsed.replace(tzinfo=None)
    except Exception:
        return None


def _extract_name_value_scalar(raw: Any) -> Any:
    if isinstance(raw, dict):
        if "value" in raw:
            return raw.get("value")
        if "name" in raw and len(raw) == 1:
            return raw.get("name")
    return raw


def _flatten_record(record: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(record, dict):
        return {}

    flattened = dict(record)
    name_value_list = flattened.get("name_value_list")
    if isinstance(name_value_list, dict):
        for key, value in name_value_list.items():
            if key not in flattened:
                flattened[key] = _extract_name_value_scalar(value)

    attrs = flattened.get("attributes")
    if isinstance(attrs, dict):
        for key, value in attrs.items():
            if key not in flattened:
                flattened[key] = _extract_name_value_scalar(value)

    return flattened


def _extract_records(value: Any) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    record_list_keys = {"records", "entry_list", "items", "results"}
    ignored_parent_keys = {
        "fields",
        "field_defs",
        "defs",
        "columns",
        "column_defs",
        "labels",
        "module_defs",
        "layout",
        "metadata",
        "menu",
    }

    def is_record_candidate(item: dict[str, Any]) -> bool:
        record_id = _safe_text(item.get("id"))
        if record_id and _UUID_RE.match(record_id):
            return True

        # Fallback for some list/search responses that omit id but still hold entity row data.
        has_person_shape = bool(_safe_text(item.get("first_name"))) and bool(_safe_text(item.get("last_name")))
        has_company_shape = bool(_safe_text(item.get("account_name") or item.get("company")))
        has_crm_dates = bool(_safe_text(item.get("date_modified") or item.get("date_entered")))
        is_field_def_shape = "vname" in item and "type" in item and ("name" in item or "rname" in item)
        return (has_person_shape and (has_company_shape or has_crm_dates)) and not is_field_def_shape

    def walk(node: Any, parent_key: str = "") -> None:
        if isinstance(node, list):
            if parent_key in record_list_keys:
                for item in node:
                    if isinstance(item, dict) and is_record_candidate(item):
                        records.append(_flatten_record(item))
            for item in node:
                walk(item, parent_key=parent_key)
            return

        if not isinstance(node, dict):
            return

        flattened = _flatten_record(node)
        if (
            parent_key in record_list_keys
            and is_record_candidate(flattened)
            and parent_key not in ignored_parent_keys
        ):
            records.append(flattened)

        for key, child in node.items():
            walk(child, parent_key=key)

    walk(value)

    deduped: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for item in records:
        record_id = _safe_text(item.get("id"))
        if record_id:
            if record_id in seen_ids:
                continue
            seen_ids.add(record_id)
        deduped.append(item)
    return deduped


def _filter_valid_contact_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    valid: list[dict[str, Any]] = []
    for row in records:
        record_id = _safe_text(row.get("id"))
        if not _UUID_RE.match(record_id):
            continue

        name = _safe_text(row.get("name"))
        first = _safe_text(row.get("first_name"))
        last = _safe_text(row.get("last_name"))
        email = _safe_text(row.get("email1"))
        phone = _safe_text(row.get("phone_mobile") or row.get("phone_work"))
        company = _safe_text(row.get("account_name") or row.get("company"))
        if not any([name, first, last, email, phone, company]):
            continue
        valid.append(row)
    return valid


def _parse_date_token(value: str) -> datetime | None:
    raw = _safe_text(value)
    if not raw:
        return None

    for fmt in ("%Y-%m-%d", "%d.%m.%Y", "%d.%m.%y"):
        try:
            return datetime.strptime(raw, fmt)
        except ValueError:
            continue
    return None


def _extract_date_range_from_text(text: str) -> tuple[datetime, datetime] | None:
    source = _safe_text(text)
    if not source:
        return None

    tokens = re.findall(r"\d{4}-\d{2}-\d{2}|\d{1,2}\.\d{1,2}\.\d{2,4}", source)
    parsed: list[datetime] = []
    for token in tokens:
        dt = _parse_date_token(token)
        if dt is not None:
            parsed.append(dt.replace(hour=0, minute=0, second=0, microsecond=0))

    if len(parsed) < 2:
        return None
    start = min(parsed)
    end = max(parsed) + timedelta(days=1)
    return start, end


def _record_name(record: dict[str, Any]) -> str:
    name = _safe_text(record.get("name"))
    if name:
        return name
    first = _safe_text(record.get("first_name"))
    last = _safe_text(record.get("last_name"))
    full = f"{first} {last}".strip()
    if full:
        return full
    return "(bez nazvu)"


def _best_record_match(records: list[dict[str, Any]], search: str) -> dict[str, Any] | None:
    if not records:
        return None
    search_norm = _normalize_text(search)
    if not search_norm:
        return None
    best: tuple[int, dict[str, Any]] | None = None
    for record in records:
        candidate = _normalize_text(_record_name(record))
        if not candidate:
            continue
        score = 0
        if candidate == search_norm:
            score = 100
        elif candidate.startswith(search_norm):
            score = 85
        elif search_norm in candidate:
            score = 75
        elif candidate in search_norm:
            score = 60
        if best is None or score > best[0]:
            best = (score, record)
    if best and best[0] >= 70:
        return best[1]
    return None


def _extract_rag_records(
    results: list[dict[str, Any]],
    *,
    module_hint: str | None = None,
) -> list[dict[str, Any]]:
    wanted = _safe_text(module_hint).lower()
    rows: list[dict[str, Any]] = []

    for item in results:
        if not isinstance(item, dict):
            continue
        payload = item.get("payload")
        if not isinstance(payload, dict):
            continue
        payload_module = _safe_text(payload.get("module")).lower()
        if wanted and payload_module != wanted:
            continue
        record = payload.get("record")
        if not isinstance(record, dict):
            continue
        row = dict(record)
        row["_module_hint"] = payload_module
        try:
            row["_rag_score"] = float(item.get("score") or 0.0)
        except Exception:
            row["_rag_score"] = 0.0
        rows.append(row)

    # Keep highest-score duplicate by id.
    deduped: dict[str, dict[str, Any]] = {}
    no_id: list[dict[str, Any]] = []
    for row in rows:
        row_id = _safe_text(row.get("id"))
        if not row_id:
            no_id.append(row)
            continue
        prev = deduped.get(row_id)
        if prev is None or float(row.get("_rag_score", 0.0)) > float(prev.get("_rag_score", 0.0)):
            deduped[row_id] = row
    return list(deduped.values()) + no_id


def _record_primary_name(record: dict[str, Any]) -> str:
    if not isinstance(record, dict):
        return ""
    direct = _safe_text(record.get("name") or record.get("NAME"))
    if direct:
        return direct
    alt = _safe_text(record.get("account_name") or record.get("company") or record.get("ACCOUNT_NAME"))
    if alt:
        return alt
    first = _safe_text(record.get("first_name") or record.get("FIRST_NAME"))
    last = _safe_text(record.get("last_name") or record.get("LAST_NAME"))
    full = f"{first} {last}".strip()
    return full


def _expand_fuzzy_token_variants(token: str) -> set[str]:
    variants = {token}
    suffixes = (
        "ovou",
        "ovou",
        "ovi",
        "ove",
        "ova",
        "ou",
        "em",
        "am",
        "um",
        "om",
        "m",
        "a",
        "u",
        "e",
        "y",
        "i",
    )
    for suffix in suffixes:
        if len(token) <= len(suffix) + 2:
            continue
        if token.endswith(suffix):
            variants.add(token[: -len(suffix)])
    return {item for item in variants if item}


def _fuzzy_tokens(value: str) -> set[str]:
    normalized = _normalize_text(value)
    if not normalized:
        return set()
    tokens = [token for token in normalized.split() if token]
    expanded: set[str] = set()
    for token in tokens:
        expanded.update(_expand_fuzzy_token_variants(token))
    return expanded


def _score_lexical_fuzzy(query: str, candidate: str) -> float:
    q = _normalize_text(query)
    c = _normalize_text(candidate)
    if not q or not c:
        return 0.0
    if q == c:
        return 1.0
    if q in c:
        return 0.96
    if c in q:
        return 0.85

    q_tokens = _fuzzy_tokens(q)
    c_tokens = _fuzzy_tokens(c)
    if not q_tokens or not c_tokens:
        return 0.0

    overlap = len(q_tokens & c_tokens) / max(1, len(q_tokens))
    ratio = SequenceMatcher(None, q, c).ratio()
    score = max(overlap * 0.95, ratio * 0.75)
    if overlap >= 0.66 and ratio >= 0.55:
        score = max(score, 0.82)
    return min(1.0, score)


def _dedupe_rag_hits_by_record(results: list[dict[str, Any]], *, limit: int) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for item in results:
        if not isinstance(item, dict):
            continue
        payload = item.get("payload")
        if not isinstance(payload, dict):
            continue
        record = payload.get("record")
        if not isinstance(record, dict):
            continue
        record_id = _safe_text(record.get("id") or payload.get("record_id"))
        if not record_id:
            continue
        prev = merged.get(record_id)
        current_score = float(item.get("score") or 0.0)
        if prev is None or current_score > float(prev.get("score") or 0.0):
            merged[record_id] = item
    deduped = sorted(merged.values(), key=lambda row: float(row.get("score") or 0.0), reverse=True)
    return deduped[: max(1, limit)]


def _lexical_module_search_in_qdrant(
    *,
    rag_service: TenantRAGService,
    tenant_id: str,
    module: str,
    query: str,
    limit: int,
) -> list[dict[str, Any]]:
    module_l = _safe_text(module).lower()
    if not module_l:
        return []

    # Keep this bounded but high enough for larger tenant collections.
    scan_cap = 120000
    scanned = 0
    offset = None
    hits: list[dict[str, Any]] = []

    collection = rag_service._tenant_collection(tenant_id)
    while True:
        points, offset = rag_service.client.scroll(
            collection_name=collection,
            scroll_filter=None,
            with_payload=True,
            with_vectors=False,
            limit=512,
            offset=offset,
        )
        if not points:
            break

        for point in points:
            scanned += 1
            payload = point.payload or {}
            payload_module = _safe_text(payload.get("module")).lower()
            if payload_module != module_l:
                continue
            record = payload.get("record")
            if not isinstance(record, dict):
                continue

            name = _record_primary_name(record)
            score = _score_lexical_fuzzy(query, name)
            if score < 0.62:
                continue
            hits.append(
                {
                    "score": score,
                    "payload": {
                        "module": payload_module,
                        "record_id": _safe_text(record.get("id") or payload.get("record_id")),
                        "record": record,
                        "source": "lexical",
                    },
                }
            )

        if offset is None or scanned >= scan_cap:
            break

    hits.sort(key=lambda row: float(row.get("score") or 0.0), reverse=True)
    return _dedupe_rag_hits_by_record(hits, limit=limit)


def _score_text_match(search: str, candidate: str) -> int:
    s = _normalize_text(search)
    c = _normalize_text(candidate)
    if not s or not c:
        return 0
    if s == c:
        return 100
    if c.startswith(s):
        return 85
    if s in c:
        return 75
    if c in s:
        return 60
    s_tokens = set(s.split())
    c_tokens = set(c.split())
    if not s_tokens:
        return 0
    overlap = len(s_tokens & c_tokens) / len(s_tokens)
    return int(overlap * 55)


def _score_account_candidate(company_name: str, record: dict[str, Any]) -> int:
    names = [
        _safe_text(record.get("name")),
        _safe_text(record.get("account_name")),
        _safe_text(record.get("company")),
    ]
    base = max((_score_text_match(company_name, name) for name in names), default=0)
    rag_boost = min(10, int(float(record.get("_rag_score", 0.0)) * 10))
    return base + rag_boost


def _select_account_candidates(
    *,
    company_name: str,
    records: list[dict[str, Any]],
    limit: int = 5,
    min_name_score: int = 70,
) -> list[dict[str, Any]]:
    scored: list[tuple[int, dict[str, Any]]] = []
    for row in records:
        row_id = _safe_text(row.get("id"))
        if not _UUID_RE.match(row_id):
            continue
        names = [
            _safe_text(row.get("name")),
            _safe_text(row.get("account_name")),
            _safe_text(row.get("company")),
        ]
        base_score = max((_score_text_match(company_name, name) for name in names), default=0)
        if base_score < min_name_score:
            continue
        score = _score_account_candidate(company_name, row)
        enriched = dict(row)
        enriched["_name_score"] = base_score
        scored.append((score, enriched))

    scored.sort(key=lambda item: item[0], reverse=True)
    if not scored:
        return []
    best_score = scored[0][0]
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for score, row in scored:
        if score < max(min_name_score, best_score - 8):
            continue
        row_id = _safe_text(row.get("id"))
        if row_id in seen:
            continue
        seen.add(row_id)
        selected.append(row)
        if len(selected) >= max(1, limit):
            break
    return selected


def _contact_row_account_ref(row: dict[str, Any]) -> tuple[str, str]:
    if not isinstance(row, dict):
        return "", ""
    account_id = _safe_text(
        row.get("account_id")
        or row.get("accounts|id")
        or row.get("account")
        or row.get("accountid")
    )
    account_name = _safe_text(
        row.get("account_name")
        or row.get("company")
        or row.get("accounts|name")
        or row.get("name")
    )
    return account_id, account_name


def _account_names_from_contact_rows(rows: list[dict[str, Any]]) -> dict[str, str]:
    names_by_id: dict[str, str] = {}
    best_score: dict[str, float] = {}
    for row in rows:
        account_id, account_name = _contact_row_account_ref(row)
        if not account_id or not account_name:
            continue
        score = float(row.get("_rag_score", 0.0))
        prev_score = best_score.get(account_id, -1.0)
        prev_name = names_by_id.get(account_id, "")
        if score > prev_score or (score == prev_score and len(account_name) > len(prev_name)):
            names_by_id[account_id] = account_name
            best_score[account_id] = score
    return names_by_id


def _enrich_account_rows_with_contact_names(
    *,
    account_rows: list[dict[str, Any]],
    contact_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not account_rows:
        return []
    names_by_id = _account_names_from_contact_rows(contact_rows)
    if not names_by_id:
        return account_rows
    enriched: list[dict[str, Any]] = []
    for row in account_rows:
        row_id = _safe_text(row.get("id"))
        name = _safe_text(row.get("name") or row.get("account_name") or row.get("company"))
        if row_id and not name and row_id in names_by_id:
            updated = dict(row)
            updated["name"] = names_by_id[row_id]
            enriched.append(updated)
            continue
        enriched.append(row)
    return enriched


def _lookup_account_names_in_contacts_by_ids(
    *,
    rag_service: TenantRAGService,
    tenant_id: str,
    account_ids: list[str],
) -> dict[str, str]:
    wanted = {row_id for row_id in account_ids if _UUID_RE.match(_safe_text(row_id))}
    if not wanted:
        return {}

    collection = rag_service._tenant_collection(tenant_id)
    offset = None
    scanned = 0
    scan_cap = 160000
    names_by_id: dict[str, str] = {}

    while True:
        points, offset = rag_service.client.scroll(
            collection_name=collection,
            scroll_filter=None,
            with_payload=True,
            with_vectors=False,
            limit=768,
            offset=offset,
        )
        if not points:
            break

        for point in points:
            scanned += 1
            payload = point.payload or {}
            module = _safe_text(payload.get("module")).lower()
            if module != "contacts":
                continue
            record = payload.get("record")
            if not isinstance(record, dict):
                continue
            account_id, account_name = _contact_row_account_ref(record)
            if not account_id or not account_name:
                continue
            if account_id not in wanted:
                continue
            prev = names_by_id.get(account_id, "")
            if len(account_name) > len(prev):
                names_by_id[account_id] = account_name

        if offset is None or scanned >= scan_cap or wanted.issubset(set(names_by_id.keys())):
            break

    return names_by_id


def _resolve_accounts_from_qdrant(
    *,
    rag_service: TenantRAGService | None,
    tenant_id: str,
    company_name: str,
    limit: int = 5,
) -> list[dict[str, Any]]:
    if rag_service is None:
        return []
    query = _safe_text(company_name)
    if not query:
        return []

    results = rag_service.search(tenant_id=tenant_id, query=query, limit=50)
    account_rows = _extract_rag_records(results, module_hint="accounts")
    contact_rows = _extract_rag_records(results, module_hint="contacts")
    account_rows = _enrich_account_rows_with_contact_names(
        account_rows=account_rows,
        contact_rows=contact_rows,
    )
    missing_ids = [
        _safe_text(row.get("id"))
        for row in account_rows
        if _UUID_RE.match(_safe_text(row.get("id"))) and not _record_primary_name(row)
    ]
    if missing_ids:
        id_lookup = _lookup_account_names_in_contacts_by_ids(
            rag_service=rag_service,
            tenant_id=tenant_id,
            account_ids=missing_ids,
        )
        if id_lookup:
            updated_rows: list[dict[str, Any]] = []
            for row in account_rows:
                row_id = _safe_text(row.get("id"))
                if row_id and row_id in id_lookup and not _record_primary_name(row):
                    row = dict(row)
                    row["name"] = id_lookup[row_id]
                updated_rows.append(row)
            account_rows = updated_rows

    # Fallback: derive account candidates from contact records when Accounts are not ingested.
    if not account_rows:
        for row in contact_rows:
            account_id, account_name = _contact_row_account_ref(row)
            if not account_id or not account_name:
                continue
            # Do not promote unrelated contact companies into account candidates.
            if _score_text_match(query, account_name) < 70:
                continue
            account_rows.append(
                {
                    "id": account_id,
                    "name": account_name,
                    "_module_hint": "accounts",
                    "_rag_score": float(row.get("_rag_score", 0.0)),
                }
            )

    return _select_account_candidates(
        company_name=query,
        records=account_rows,
        limit=limit,
        min_name_score=70,
    )


def _query_requires_filled_email(text: str) -> bool:
    normalized = _normalize_text(text)
    return any(
        token in normalized
        for token in (
            "email",
            "e mail",
            "mail",
            "s emailem",
            "vyplnenym emailem",
            "filled email",
            "with email",
        )
    )


def _build_contacts_filter_by_account_ids(
    *,
    account_ids: list[str],
    require_email: bool = False,
) -> dict[str, Any]:
    account_operands = [{"field": "id", "type": "eq", "value": row_id} for row_id in account_ids if row_id]
    relation_operator = "and" if len(account_operands) <= 1 else "or"
    relation_filter: dict[str, Any] = {
        "operator": relation_operator,
        "operands": account_operands,
    }
    filter_operands: list[dict[str, Any]] = [
        {
            "operator": "and",
            "operands": [
                {
                    "module": "Accounts",
                    "type": "relate",
                    "name": "account_name",
                    "relationship": ["accounts"],
                    "filter": relation_filter,
                }
            ],
        }
    ]
    if require_email:
        filter_operands.append({"field": "email", "type": "nnull", "value": None})
    return {"operator": "and", "operands": filter_operands}


def _crm_detail_link(module: str, record_id: str) -> str:
    clean_module = _safe_text(module) or "Home"
    clean_id = _safe_text(record_id)
    return f"/#detail/{clean_module}/{clean_id}" if clean_id else "/#"


def _format_date(value: Any) -> str:
    dt = _parse_datetime(value)
    if dt is None:
        return _safe_text(value)
    return dt.strftime("%d.%m.%Y %H:%M")


def _non_empty_meta(meta: dict[str, Any]) -> dict[str, Any]:
    clean: dict[str, Any] = {}
    for key, value in meta.items():
        text = _safe_text(value)
        if text:
            clean[key] = text
    return clean


def _record_card(module: str, title: str, record_id: str, meta: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": f"{module}-{record_id}" if record_id else f"{module}-row",
        "type": "record",
        "title": title,
        "tag": module,
        "meta": _non_empty_meta(meta),
        "actions": [
            {
                "label": "Otevrit v CRM",
                "intent": "primary",
                "action": "link",
                "url": _crm_detail_link(module, record_id),
            }
        ]
        if record_id
        else [],
    }


def _table_card(
    *,
    title: str,
    tag: str,
    columns: list[dict[str, str]],
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "id": f"table-{_normalize_text(title) or 'crm'}",
        "type": "table",
        "title": title,
        "tag": tag,
        "columns": columns,
        "rows": rows,
    }


def _extract_person_name(query: str, context: dict[str, Any] | None) -> str:
    patterns = [
        r"\bkdo\s+je\s+([^?.!,]+)",
        r"\bkontakt\s+([^?.!,]+)",
        r"\bs\s+([^?.!,]+?)(?:\s+na\s+|\s+v\s+|$)",
    ]
    for pattern in patterns:
        match = re.search(pattern, query, re.IGNORECASE)
        if match:
            return _safe_text(match.group(1)).strip(" '\"")

    entities = context.get("entities") if isinstance(context, dict) else {}
    if isinstance(entities, dict):
        candidate = _safe_text(entities.get("contact_name"))
        if candidate:
            return candidate
    return _safe_text(query)


def _recent_contacts_from_context(context: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(context, dict):
        return []
    raw = context.get("recent_contacts")
    if not isinstance(raw, list):
        return []

    contacts: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        record_id = _safe_text(item.get("id") or item.get("contact_id"))
        name = _safe_text(item.get("name") or item.get("full_name") or item.get("title"))
        first_name = _safe_text(item.get("first_name"))
        last_name = _safe_text(item.get("last_name"))
        if not name and (first_name or last_name):
            name = f"{first_name} {last_name}".strip()
        if not name:
            continue
        contact = {
            "id": record_id,
            "name": name,
            "first_name": first_name,
            "last_name": last_name,
            "account_name": _safe_text(item.get("account_name") or item.get("company")),
            "email1": _safe_text(item.get("email1") or item.get("email")),
            "phone_mobile": _safe_text(item.get("phone_mobile") or item.get("phone_work") or item.get("phone")),
        }
        contacts.append(contact)
    return contacts


def _best_recent_contact_match(query: str, candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    search = _safe_text(query)
    if not search or not candidates:
        return None

    best: tuple[float, dict[str, Any]] | None = None
    for row in candidates:
        candidate_name = _record_name(row)
        if not candidate_name:
            continue
        lexical = _score_lexical_fuzzy(search, candidate_name) * 100.0
        token_score = float(_score_text_match(search, candidate_name))
        score = max(lexical, token_score)
        if best is None or score > best[0]:
            best = (score, row)

    if best and best[0] >= 70.0:
        return best[1]
    return None


def _extract_company_name(query: str, context: dict[str, Any] | None) -> str:
    patterns = [
        r"\b(?:k|ke)\s+firme\s+([^?.!,]+)",
        r"\b(?:k|ke)\s+firm[eě]\s+([^?.!,]+)",
        r"\bspolecnosti\s+([^?.!,]+)",
        r"\bfirma\s+([^?.!,]+)",
        r"\bfirmy\s+([^?.!,]+)",
        r"\bcompany\s+([^?.!,]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, query, re.IGNORECASE)
        if match:
            return _safe_text(match.group(1)).strip(" '\"")

    entities = context.get("entities") if isinstance(context, dict) else {}
    if isinstance(entities, dict):
        candidate = _safe_text(entities.get("account_name"))
        if candidate:
            return candidate
    return _safe_text(query)


def _collect_account_ids_from_value(value: Any, sink: list[str]) -> None:
    if value is None:
        return
    if isinstance(value, str):
        candidate = _safe_text(value)
        if _UUID_RE.match(candidate):
            sink.append(candidate)
            return
        # Also support free-form LLM expressions like:
        # "AccountId == '6c3284e1-176c-ed80-77f9-652d0ad83b5c'"
        for match in re.findall(
            r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}",
            candidate,
        ):
            sink.append(match)
        return
    if isinstance(value, list):
        for item in value:
            _collect_account_ids_from_value(item, sink)
        return
    if not isinstance(value, dict):
        return

    for key in ("Accounts.id", "accounts.id", "account_id", "accounts|id", "id"):
        if key in value:
            _collect_account_ids_from_value(value.get(key), sink)

    field_name = _safe_text(value.get("field") or value.get("name")).lower()
    field_type = _safe_text(value.get("type")).lower()
    if field_name in {"id", "accounts.id", "account_id", "accounts|id"} and field_type in {"", "eq", "in"}:
        _collect_account_ids_from_value(value.get("value"), sink)

    for child in value.values():
        _collect_account_ids_from_value(child, sink)


def _extract_account_ids_from_query(
    query: Any,
    context: dict[str, Any] | None,
) -> list[str]:
    ids: list[str] = []
    _collect_account_ids_from_value(query, ids)

    if isinstance(context, dict):
        ctx_module = _safe_text(context.get("module")).lower()
        ctx_record = _safe_text(context.get("record"))
        if ctx_module == "accounts" and _UUID_RE.match(ctx_record):
            ids.append(ctx_record)

    deduped: list[str] = []
    seen: set[str] = set()
    for row_id in ids:
        clean = _safe_text(row_id)
        if not _UUID_RE.match(clean) or clean in seen:
            continue
        seen.add(clean)
        deduped.append(clean)
    return deduped


def _infer_data_query_type(query: str, requested: str = "auto") -> str:
    explicit = _safe_text(requested).lower()
    normalized = _normalize_text(query)
    has_contact = any(token in normalized for token in ("kontakt", "contacts", "contact"))
    has_company = any(token in normalized for token in ("firma", "spolecnost", "company", "firmy", "firme"))

    # LLM often sends query_type="contacts" even when user asked contacts by company.
    # Prefer user-intent heuristic over broad explicit aliases.
    if explicit in {"contacts", "contact"} and has_contact and has_company:
        return "contacts_by_company"

    explicit_aliases = {
        "meetings_next_week": "meetings_next_week",
        "meetings_range": "meetings_range",
        "contact": "contact_by_name",
        "contacts": "contact_by_name",
        "contact_by_name": "contact_by_name",
        "contacts_by_company": "contacts_by_company",
        "company_contacts": "contacts_by_company",
        "search": "search",
        "generic": "generic_search",
        "generic_search": "generic_search",
    }
    if explicit in explicit_aliases:
        return explicit_aliases[explicit]
    if explicit in {"meetings", "meeting", "schuzky", "schuzka"}:
        normalized_explicit_query = _normalize_text(query)
        if (
            "pristi tyden" in normalized_explicit_query
            or "dalsi tyden" in normalized_explicit_query
            or "next week" in normalized_explicit_query
        ):
            return "meetings_next_week"
        return "meetings_range"
    if explicit in {"meetings_next_week", "contact_by_name", "contacts_by_company", "generic_search"}:
        return explicit

    if "kdo je" in normalized:
        return "contact_by_name"

    has_meeting = any(token in normalized for token in ("schuzk", "meeting"))
    has_next_week = (
        "pristi tyden" in normalized
        or "dalsi tyden" in normalized
        or "next week" in normalized
    )
    if has_meeting and has_next_week:
        return "meetings_next_week"
    if has_meeting and ("obdobi" in normalized or "od " in normalized or "do " in normalized):
        return "meetings_range"
    if has_meeting and _extract_date_range_from_text(query):
        return "meetings_range"

    if has_contact and has_company:
        return "contacts_by_company"

    if has_contact:
        return "contact_by_name"

    return "generic_search"


def _filter_records_by_company(
    records: list[dict[str, Any]],
    company_name: str,
    *,
    fallback_to_original: bool = True,
) -> list[dict[str, Any]]:
    normalized_company = _normalize_text(company_name)
    if not normalized_company:
        return records

    filtered: list[dict[str, Any]] = []
    for record in records:
        account = _normalize_text(_safe_text(record.get("account_name") or record.get("company")))
        if normalized_company in account or account in normalized_company:
            filtered.append(record)
    if filtered:
        return filtered
    return records if fallback_to_original else []


def _dedupe_and_limit(records: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for record in records:
        record_id = _safe_text(record.get("id"))
        name = _normalize_text(_record_name(record))
        fingerprint = f"{record_id}|{name}"
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        deduped.append(record)
        if len(deduped) >= limit:
            break
    return deduped


def _merge_records_by_id(*record_sets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen: set[str] = set()
    for rows in record_sets:
        for row in rows:
            rec_id = _safe_text(row.get("id"))
            if not rec_id or rec_id in seen:
                continue
            seen.add(rec_id)
            merged.append(row)
    return merged


def _meeting_cards(
    records: list[dict[str, Any]],
    total_count: int,
    *,
    force_table: bool = False,
) -> list[dict[str, Any]]:
    if not records:
        return []
    if total_count <= 4 and not force_table:
        cards = []
        for row in records:
            cards.append(
                _record_card(
                    "Meetings",
                    _record_name(row),
                    _safe_text(row.get("id")),
                    {
                        "Datum": _format_date(row.get("date_start")),
                        "Misto": _safe_text(row.get("location")),
                        "Stav": _safe_text(row.get("status")),
                    },
                )
            )
        return cards

    rows: list[dict[str, Any]] = []
    for row in records:
        rec_id = _safe_text(row.get("id"))
        rows.append(
            {
                "id": rec_id,
                "cells": {
                    "date_start": _format_date(row.get("date_start")),
                    "name": _record_name(row),
                    "location": _safe_text(row.get("location")),
                    "status": _safe_text(row.get("status")),
                },
                "link": {
                    "label": "Detail",
                    "url": _crm_detail_link("Meetings", rec_id),
                },
            }
        )

    return [
        _table_card(
            title=f"Schuzky ({total_count})",
            tag="Meetings",
            columns=[
                {"key": "date_start", "label": "Datum"},
                {"key": "name", "label": "Nazev"},
                {"key": "location", "label": "Misto"},
                {"key": "status", "label": "Stav"},
            ],
            rows=rows,
        )
    ]


def _contact_cards(
    records: list[dict[str, Any]],
    total_count: int,
    title: str,
    *,
    force_table: bool = False,
) -> list[dict[str, Any]]:
    if not records:
        return []

    if total_count <= 4 and not force_table:
        cards = []
        for row in records:
            cards.append(
                _record_card(
                    "Contacts",
                    _record_name(row),
                    _safe_text(row.get("id")),
                    {
                        "Firma": _safe_text(row.get("account_name") or row.get("company")),
                        "E-mail": _safe_text(row.get("email1")),
                        "Telefon": _safe_text(row.get("phone_mobile") or row.get("phone_work")),
                    },
                )
            )
        return cards

    rows: list[dict[str, Any]] = []
    for row in records:
        rec_id = _safe_text(row.get("id"))
        rows.append(
            {
                "id": rec_id,
                "cells": {
                    "name": _record_name(row),
                    "company": _safe_text(row.get("account_name") or row.get("company")),
                    "email": _safe_text(row.get("email1")),
                    "phone": _safe_text(row.get("phone_mobile") or row.get("phone_work")),
                },
                "link": {
                    "label": "Detail",
                    "url": _crm_detail_link("Contacts", rec_id),
                },
            }
        )

    return [
        _table_card(
            title=title,
            tag="Contacts",
            columns=[
                {"key": "name", "label": "Jmeno"},
                {"key": "company", "label": "Firma"},
                {"key": "email", "label": "E-mail"},
                {"key": "phone", "label": "Telefon"},
            ],
            rows=rows,
        )
    ]


def _generic_cards(records: list[dict[str, Any]], total_count: int) -> list[dict[str, Any]]:
    if not records:
        return []
    if total_count <= 4:
        cards = []
        for row in records:
            module = _safe_text(row.get("_module_hint")) or "CRM"
            cards.append(
                _record_card(
                    module,
                    _record_name(row),
                    _safe_text(row.get("id")),
                    {
                        "Modul": module,
                        "Stav": _safe_text(row.get("status")),
                        "Upraveno": _format_date(row.get("date_modified")),
                    },
                )
            )
        return cards

    rows: list[dict[str, Any]] = []
    for row in records:
        module = _safe_text(row.get("_module_hint")) or "CRM"
        rec_id = _safe_text(row.get("id"))
        rows.append(
            {
                "id": f"{module}-{rec_id}" if rec_id else module,
                "cells": {
                    "module": module,
                    "name": _record_name(row),
                    "status": _safe_text(row.get("status")),
                },
                "link": {
                    "label": "Detail",
                    "url": _crm_detail_link(module, rec_id),
                }
                if rec_id
                else None,
            }
        )

    return [
        _table_card(
            title=f"Vysledky ({total_count})",
            tag="CRM",
            columns=[
                {"key": "module", "label": "Modul"},
                {"key": "name", "label": "Nazev"},
                {"key": "status", "label": "Stav"},
            ],
            rows=rows,
        )
    ]


def _next_week_range(now: datetime | None = None) -> tuple[datetime, datetime]:
    current = now or datetime.now()
    days_to_next_monday = 7 - current.weekday()
    start = (current + timedelta(days=days_to_next_monday)).replace(
        hour=0,
        minute=0,
        second=0,
        microsecond=0,
    )
    end = start + timedelta(days=7)
    return start, end


def _build_meetings_date_filter(range_start: datetime, range_end_exclusive: datetime) -> dict[str, Any]:
    end_inclusive = (range_end_exclusive - timedelta(days=1)).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    return {
        "operator": "and",
        "operands": [
            {
                "operator": "and",
                "operands": [
                    {
                        "field": "date_start",
                        "fieldModule": None,
                        "fieldRel": None,
                        "type": "moreThanInclude",
                        "value": range_start.strftime("%Y-%m-%d"),
                        "relationField": None,
                    },
                    {
                        "field": "date_start",
                        "fieldModule": None,
                        "fieldRel": None,
                        "type": "lessThanInclude",
                        "value": end_inclusive.strftime("%Y-%m-%d"),
                        "relationField": None,
                    },
                ],
            }
        ],
    }


def _format_filter_datetime_boundary(value: str) -> str:
    raw = _safe_text(value)
    dt = _parse_datetime(raw)
    if dt is None:
        return ""
    if re.search(r"\d{1,2}:\d{2}", raw):
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    return dt.strftime("%Y-%m-%d")


def _build_login_user_meetings_window_filter(date_from: str, date_to: str) -> dict[str, Any]:
    date_from_value = _format_filter_datetime_boundary(date_from)
    date_to_value = _format_filter_datetime_boundary(date_to)
    return {
        "operator": "and",
        "operands": [
            {
                "field": "assigned_user_id",
                "type": "eq",
                "value": "{%LOGIN_USER%}",
            },
            {
                "field": "date_start",
                "fieldModule": None,
                "fieldRel": None,
                "type": "moreThanInclude",
                "value": date_from_value,
                "relationField": None,
            },
            {
                "field": "date_start",
                "fieldModule": None,
                "fieldRel": None,
                "type": "lessThanInclude",
                "value": date_to_value,
                "relationField": None,
            },
        ],
    }


def _sort_records_by_datetime(records: list[dict[str, Any]], field_name: str) -> list[dict[str, Any]]:
    return sorted(
        records,
        key=lambda row: _parse_datetime(row.get(field_name)) or datetime.max,
    )


def build_tools(
    *,
    tenant_id: str,
    user_id: str,
    input_text: str,
    request_context: dict[str, Any] | None,
    crm_client: SugarClient,
    rag_service: TenantRAGService | None,
    action_confirmation: bool = False,
) -> list:
    settings = get_settings()
    write_service = CRMWriteService(
        tenant_id=tenant_id,
        user_id=user_id,
        crm_client=crm_client,
        input_text=input_text,
        request_context=request_context,
        rag_service=rag_service,
        action_confirmation=action_confirmation,
    )
    read_helpers = CRMReadHelpers(
        safe_text=_safe_text,
        normalize_text=_normalize_text,
        parse_datetime=_parse_datetime,
        extract_records=_extract_records,
        extract_date_range_from_text=_extract_date_range_from_text,
        infer_data_query_type=_infer_data_query_type,
        next_week_range=lambda: _next_week_range(),
        build_meetings_date_filter=_build_meetings_date_filter,
        sort_records_by_datetime=_sort_records_by_datetime,
        dedupe_and_limit=_dedupe_and_limit,
        meeting_cards=_meeting_cards,
        filter_valid_contact_records=_filter_valid_contact_records,
        contact_cards=lambda records, total_count, title: _contact_cards(
            records,
            total_count,
            title=title,
            force_table=total_count > 4,
        ),
        contact_cards_force_table=lambda records, total_count, title: _contact_cards(
            records,
            total_count,
            title=title,
            force_table=True,
        ),
        generic_cards=_generic_cards,
        extract_person_name=_extract_person_name,
        recent_contacts_from_context=_recent_contacts_from_context,
        best_recent_contact_match=_best_recent_contact_match,
        extract_rag_records=lambda results, module_hint: _extract_rag_records(
            results,
            module_hint=module_hint,
        ),
        merge_records_by_id=_merge_records_by_id,
        best_record_match=_best_record_match,
        extract_company_name=_extract_company_name,
        extract_account_ids_from_query=_extract_account_ids_from_query,
        resolve_accounts_from_qdrant=lambda company_name: _resolve_accounts_from_qdrant(
            rag_service=rag_service,
            tenant_id=tenant_id,
            company_name=company_name,
            limit=5,
        ),
        select_account_candidates=lambda company_name, records, limit, min_name_score: _select_account_candidates(
            company_name=company_name,
            records=records,
            limit=limit,
            min_name_score=min_name_score,
        ),
        query_requires_filled_email=_query_requires_filled_email,
        build_contacts_filter_by_account_ids=lambda account_ids, require_email: _build_contacts_filter_by_account_ids(
            account_ids=account_ids,
            require_email=require_email,
        ),
        filter_records_by_company=lambda records, company_name, fallback_to_original: _filter_records_by_company(
            records,
            company_name,
            fallback_to_original=fallback_to_original,
        ),
    )
    read_service = CRMReadService(
        tenant_id=tenant_id,
        user_id=user_id,
        input_text=input_text,
        request_context=request_context,
        crm_client=crm_client,
        rag_service=rag_service,
        helpers=read_helpers,
    )

    @tool("crm_action_tool")
    async def crm_action_tool(module: str, action: str, data_json: str = "{}") -> str:
        """
        Create, update, or delete CRM records. Use ONLY for write operations.

        module: Meetings | Calls | Tasks | Notes | Contacts | Accounts | Leads
        action: create | update | delete

        data_json: JSON object with the record fields.

        Creating a Meeting:
          module="Meetings", action="create",
          data_json='{"name":"...", "date_start":"YYYY-MM-DD HH:MM:SS",
                      "duration_hours":1, "status":"Planned",
                      "invitees":[{"type":"Contacts","id":"<uuid>"}]}'

        Logging a Call:
          module="Calls", action="create",
          data_json='{"name":"...", "direction":"Outbound", "status":"Held",
                      "duration_hours":0, "duration_minutes":15}'

        Creating a Task:
          module="Tasks", action="create",
          data_json='{"name":"...", "date_due":"YYYY-MM-DD", "status":"Not Started",
                      "contact_id":"<uuid>"}'

        Updating a record:
          module="Contacts", action="update",
          data_json='{"id":"<uuid>", "phone_mobile":"+420..."}'

        ALWAYS resolve contact/account IDs first using crm_query_tool or
        rag_search_tool before creating related records.
        When action_confirmation is enabled, mutations return a pending_action
        envelope — do not re-submit; wait for user confirmation.
        """
        if settings.crm_mode.lower() == "off":
            return json.dumps({"status": "crm-disabled",
                               "message": "CRM mode is off."}, ensure_ascii=False)

        async with ToolCallLogger(
            "crm_action_tool", tenant_id, user_id,
            inputs={"module": module, "action": action, "data_json": data_json},
        ) as tcl:
            try:
                result = await write_service.execute_action(
                    module=module, action=action, data_json=data_json
                )
            except Exception as exc:
                error_payload = {"status": "error", "message": str(exc)}
                tcl.set_output(error_payload)
                return json.dumps(error_payload, ensure_ascii=False)
            tcl.set_output({"status": result.get("status"), "module": module, "action": action})
            return json.dumps(result, ensure_ascii=False)

    @tool("rag_search_tool")
    async def rag_search_tool(query: str, limit: int = 5, module: str = "") -> str:
        """
        Semantic vector search across tenant knowledge in Qdrant.

        Use this BEFORE crm_query_tool when you need to resolve a name to a UUID,
        or when the user's query is vague and you need to find the best matching
        record to anchor subsequent queries.

        query: Natural language search — names, company names, topics
        limit: Number of results (default 5)
        module: Optional filter: contacts | accounts | meetings | calls | tasks

        Returns records with IDs you can pass to crm_query_tool filters or
        crm_action_tool data_json.
        """
        async with ToolCallLogger(
            "rag_search_tool", tenant_id, user_id,
            inputs={"query": query, "limit": limit, "module": module},
        ) as tcl:
            if rag_service is None:
                result_payload = {"warning": "RAG unavailable", "results": []}
                tcl.set_output(result_payload)
                return json.dumps(result_payload, ensure_ascii=False)
            module_filter = _safe_text(module).lower()
            account_lookup_query = (
                _normalize_account_lookup_query(query)
                if module_filter == "accounts"
                else query
            )

            wide_limit = max(10, int(limit or 5) * 4)
            results = rag_service.search(tenant_id=tenant_id, query=account_lookup_query, limit=wide_limit)
            if module_filter:
                filtered: list[dict[str, Any]] = []
                for item in results:
                    if not isinstance(item, dict):
                        continue
                    payload = item.get("payload")
                    if not isinstance(payload, dict):
                        continue
                    payload_module = _safe_text(payload.get("module")).lower()
                    if payload_module == module_filter:
                        filtered.append(item)
                results = filtered
    
                # Augment vector search with lexical fuzzy scan (case/diacritics/morphology).
                lexical_hits = _lexical_module_search_in_qdrant(
                    rag_service=rag_service,
                    tenant_id=tenant_id,
                    module=module_filter,
                    query=account_lookup_query,
                    limit=max(10, int(limit or 5) * 4),
                )
                results = _dedupe_rag_hits_by_record(results + lexical_hits, limit=max(10, int(limit or 5) * 4))
    
                # Useful fallback for company lookups: derive account candidates from contacts.
                if module_filter == "accounts":
                    wide_results = rag_service.search(
                        tenant_id=tenant_id,
                        query=account_lookup_query,
                        limit=max(30, limit * 8),
                    )
                    contact_rows = _extract_rag_records(wide_results, module_hint="contacts")
                    derived_accounts: list[dict[str, Any]] = []
                    for row in contact_rows:
                        account_id, account_name = _contact_row_account_ref(row)
                        if not account_id or not account_name:
                            continue
                        if _score_text_match(account_lookup_query, account_name) < 65:
                            continue
                        derived_accounts.append(
                            {
                                "id": account_id,
                                "name": account_name,
                                "_rag_score": float(row.get("_rag_score", 0.0)),
                            }
                        )
                    picked = _select_account_candidates(
                        company_name=account_lookup_query,
                        records=derived_accounts,
                        limit=limit,
                        min_name_score=65,
                    )
                    derived_hits = [
                        {
                            "score": float(item.get("_rag_score", 0.0)),
                            "payload": {
                                "module": "accounts",
                                "record_id": item.get("id"),
                                "record": item,
                                "derived_from": "contacts",
                            },
                        }
                        for item in picked
                    ]
                    results = _dedupe_rag_hits_by_record(
                        results + derived_hits,
                        limit=max(10, int(limit or 5) * 4),
                    )
    
                    # CRM-backed fallback for account names when Qdrant account payloads are sparse.
                    try:
                        query_variants: list[str] = []
                        normalized_query = _normalize_text(account_lookup_query)
                        for value in (account_lookup_query, normalized_query):
                            candidate = _safe_text(value)
                            if candidate and candidate not in query_variants:
                                query_variants.append(candidate)
                        tokens = [token for token in normalized_query.split() if token]
                        for token in tokens:
                            if len(token) >= 3 and token not in query_variants:
                                query_variants.append(token)
                        query_variants = query_variants[:4]
    
                        crm_hits_by_id: dict[str, dict[str, Any]] = {}
                        for variant in query_variants:
                            crm_accounts = await crm_client.execute_module_action(
                                module="Accounts",
                                action="list",
                                data={
                                    "limit": max(10, int(limit or 5) * 4),
                                    "offset": 0,
                                    "columns": [{"field": "name", "module": "Accounts", "width": "20%", "function": None}],
                                    "filter": {
                                        "operator": "and",
                                        "operands": [
                                            {
                                                "operator": "and",
                                                "operands": [
                                                    {
                                                        "field": "*",
                                                        "fieldModule": None,
                                                        "fieldRel": None,
                                                        "type": "cont",
                                                        "value": variant,
                                                        "relationField": None,
                                                    }
                                                ],
                                            }
                                        ],
                                    },
                                    "include_field_names": False,
                                    "response_fields": ["id", "name", "account_name"],
                                },
                            )
                            for row in _extract_records(crm_accounts):
                                row_id = _safe_text(row.get("id"))
                                row_name = _record_primary_name(row)
                                if not row_id or not row_name:
                                    continue
                                score = _score_lexical_fuzzy(account_lookup_query, row_name)
                                if score < 0.55:
                                    continue
                                existing = crm_hits_by_id.get(row_id)
                                if existing is None or float(existing.get("score") or 0.0) < score:
                                    crm_hits_by_id[row_id] = {
                                        "score": max(0.72, score),
                                        "payload": {
                                            "module": "accounts",
                                            "record_id": row_id,
                                            "record": {"id": row_id, "name": row_name},
                                            "source": "crm-filter",
                                        },
                                    }
                        crm_hits = list(crm_hits_by_id.values())
                        if crm_hits:
                            results = _dedupe_rag_hits_by_record(
                                results + crm_hits,
                                limit=max(10, int(limit or 5) * 4),
                            )
                    except Exception:
                        pass
    
                # If Accounts records in Qdrant have only {"id"}, enrich names from CRM.
                if module_filter == "accounts" and results:
                    account_ids: list[str] = []
                    for item in results:
                        payload = item.get("payload") if isinstance(item, dict) else None
                        if not isinstance(payload, dict):
                            continue
                        record = payload.get("record")
                        if not isinstance(record, dict):
                            continue
                        name = _record_primary_name(record)
                        record_id = _safe_text(record.get("id") or payload.get("record_id"))
                        if record_id and not name:
                            account_ids.append(record_id)
    
                    account_ids = list(dict.fromkeys(account_ids))[:30]
                    if account_ids:
                        try:
                            id_filter = {
                                "operator": "and",
                                "operands": [
                                    {
                                        "operator": "or",
                                        "operands": [
                                            {"field": "id", "type": "eq", "value": row_id}
                                            for row_id in account_ids
                                        ],
                                    }
                                ],
                            }
                            hydrated = await crm_client.execute_module_action(
                                module="Accounts",
                                action="list",
                                data={
                                    "limit": max(30, len(account_ids) * 2),
                                    "offset": 0,
                                    "filter": id_filter,
                                    "include_field_names": False,
                                    "response_fields": ["id", "name", "account_name"],
                                },
                            )
                            names_by_id: dict[str, str] = {}
                            for row in _extract_records(hydrated):
                                row_id = _safe_text(row.get("id"))
                                row_name = _record_primary_name(row)
                                if row_id and row_name:
                                    names_by_id[row_id] = row_name
    
                            if names_by_id:
                                for item in results:
                                    payload = item.get("payload") if isinstance(item, dict) else None
                                    if not isinstance(payload, dict):
                                        continue
                                    record = payload.get("record")
                                    if not isinstance(record, dict):
                                        continue
                                    row_id = _safe_text(record.get("id") or payload.get("record_id"))
                                    if row_id and row_id in names_by_id and not _record_primary_name(record):
                                        record = dict(record)
                                        record["name"] = names_by_id[row_id]
                                        payload["record"] = record
                        except Exception:
                            # Keep raw RAG hits when hydration is not available.
                            pass
    
                    # Fallback for tests/dev where Accounts vectors may contain only IDs and CRM auth
                    # is not configured: infer account names from contact records in Qdrant.
                    unresolved_ids: list[str] = []
                    for item in results:
                        payload = item.get("payload") if isinstance(item, dict) else None
                        if not isinstance(payload, dict):
                            continue
                        record = payload.get("record")
                        if not isinstance(record, dict):
                            continue
                        row_id = _safe_text(record.get("id") or payload.get("record_id"))
                        if row_id and not _record_primary_name(record):
                            unresolved_ids.append(row_id)
                    unresolved_ids = list(dict.fromkeys(unresolved_ids))
    
                    if unresolved_ids:
                        names_by_id = _lookup_account_names_in_contacts_by_ids(
                            rag_service=rag_service,
                            tenant_id=tenant_id,
                            account_ids=unresolved_ids,
                        )
                        contact_scan = rag_service.search(
                            tenant_id=tenant_id,
                            query=query,
                            limit=max(100, int(limit or 5) * 20),
                        )
                        contact_rows = _extract_rag_records(contact_scan, module_hint="contacts")
                        names_by_id.update(_account_names_from_contact_rows(contact_rows))
                        if names_by_id:
                            for item in results:
                                payload = item.get("payload") if isinstance(item, dict) else None
                                if not isinstance(payload, dict):
                                    continue
                                record = payload.get("record")
                                if not isinstance(record, dict):
                                    continue
                                row_id = _safe_text(record.get("id") or payload.get("record_id"))
                                if row_id and row_id in names_by_id and not _record_primary_name(record):
                                    record = dict(record)
                                    record["name"] = names_by_id[row_id]
                                    payload["record"] = record
            tcl.set_output({"result_count": len(results[: max(1, int(limit or 5))]), "module_filter": _safe_text(module).lower()})
            return json.dumps(results[: max(1, int(limit or 5))], ensure_ascii=False)

    @tool("my_meetings_tool")
    async def my_meetings_tool(
        date_from: str,
        date_to: str,
        limit: int = 100,
    ) -> str:
        """
        Returns current logged-in user's meetings in a date window.

        date_from/date_to: timeframe boundaries (ISO date: YYYY-MM-DD).
        Uses CRM filter with assigned_user_id = {%LOGIN_USER%}.

        This tool is intended for:
        - displaying "my meetings" in a period
        - checking time-window conflicts before creating a new meeting
        """
        if settings.crm_mode.lower() == "off":
            return json.dumps({"status": "crm-disabled", "message_to_user": "CRM je vypnuté."}, ensure_ascii=False)

        from_boundary = _format_filter_datetime_boundary(date_from)
        to_boundary = _format_filter_datetime_boundary(date_to)
        if not from_boundary or not to_boundary:
            return json.dumps(
                {
                    "status": "error",
                    "message": "Invalid date_from/date_to. Use YYYY-MM-DD.",
                    "cards": [],
                },
                ensure_ascii=False,
            )

        from_dt = _parse_datetime(from_boundary)
        to_dt = _parse_datetime(to_boundary)
        if from_dt is None or to_dt is None or from_dt > to_dt:
            return json.dumps(
                {
                    "status": "error",
                    "message": "Invalid timeframe: date_from must be <= date_to.",
                    "cards": [],
                },
                ensure_ascii=False,
            )

        crm_filter = _build_login_user_meetings_window_filter(from_boundary, to_boundary)
        payload = {
            "limit": max(1, min(int(limit), 500)),
            "offset": 0,
            "filter": crm_filter,
            "include_field_names": False,
            "response_fields": [
                "id",
                "name",
                "date_start",
                "status",
                "location",
                "assigned_user_name",
                "parent_name",
            ],
            "order": [{"field": "date_start", "sort": "ASC", "module": "Meetings"}],
        }

        async with ToolCallLogger(
            "my_meetings_tool",
            tenant_id,
            user_id,
            inputs={"date_from": date_from, "date_to": date_to, "limit": limit},
        ) as tcl:
            try:
                raw = await crm_client.execute_module_action(
                    module="Meetings",
                    action="list",
                    data=payload,
                )
            except Exception as exc:
                error_payload = {"status": "error", "message": str(exc), "cards": []}
                tcl.set_output(error_payload)
                return json.dumps(error_payload, ensure_ascii=False)

            records = _extract_records(raw)
            total = len(records)
            cards = _meeting_cards(records, total, force_table=True)

            summary_lines: list[str] = []
            for row in records[:15]:
                summary_lines.append(
                    f"• {_format_date(row.get('date_start'))} — {_record_name(row)} ({_safe_text(row.get('status'))})"
                )

            result = {
                "status": "ok",
                "module": "Meetings",
                "date_from": from_boundary,
                "date_to": to_boundary,
                "total": total,
                "summary": "\n".join(summary_lines) if summary_lines else "Žádné záznamy.",
                "cards": cards,
            }
            tcl.set_output({"total": total, "module": "Meetings"})
            return json.dumps(result, ensure_ascii=False)

    @tool("crm_query_tool")
    async def crm_query_tool(
        module: str,
        search: str | None = None,
        filters: str = "[]",
        date_from: str | None = None,
        date_to: str | None = None,
        order_by: str | None = None,
        limit: int = 20,
    ) -> str:
        """
        Read and search CRM records. Use this for ALL read operations.

        module: Which CRM module to query.
          Contacts | Accounts | Meetings | Calls | Tasks | Notes | Leads

        search: Free-text search across all fields (names, subjects, etc).
          Use for: "find Jan Novak", "search quarterly review meetings"

        filters: JSON list of field conditions (AND-ed together).
          Each item: {"field": "<field_name>", "op": "<op>", "value": "<value>"}
          Operators: eq, neq, cont, starts, nnull, null, gte, lte
          Examples:
            [{"field": "status", "op": "eq", "value": "Held"}]
            [{"field": "account_id", "op": "eq", "value": "<uuid>"}]
            [{"field": "assigned_user_id", "op": "eq", "value": "<uuid>"},
             {"field": "status", "op": "neq", "value": "Not Started"}]

        date_from: ISO date "YYYY-MM-DD". Lower bound (inclusive).
          Filters date_start for Meetings/Calls, date_due for Tasks.
        date_to: ISO date "YYYY-MM-DD". Upper bound (inclusive).

        order_by: Sort field and direction: "date_start:asc" or "name:desc"

        limit: Max records to return (default 20, max 100)

        IMPORTANT — When to use filters vs search:
          - Use `search` when you have a name or keyword and no ID yet.
          - For Contacts, `account_id` in filters is treated as an alias and converted
            to an Accounts relation filter (uses `fieldRel=["accounts"]` and a
            `type="relate"` fallback for Coripo compatibility).
          - For contacts at a company: first resolve account_id via rag_search_tool,
            then call crm_query_tool(module="Contacts",
                                     filters='[{"field":"account_id","op":"eq","value":"<uuid>"}]')

        Do NOT use for creating, updating, or deleting — use crm_action_tool for that.
        """
        if settings.crm_mode.lower() == "off":
            return json.dumps({"status": "crm-disabled",
                               "message_to_user": "CRM je vypnuté."}, ensure_ascii=False)

        async with ToolCallLogger(
            "crm_query_tool", tenant_id, user_id,
            inputs={"module": module, "search": search, "filters": filters,
                    "date_from": date_from, "date_to": date_to,
                    "order_by": order_by, "limit": limit},
        ) as tcl:
            try:
                parsed_filters = [FilterSpec(**f) for f in json.loads(filters or "[]")]
            except Exception as exc:
                error_payload = {"status": "error",
                                 "message": f"Invalid filters JSON: {exc}",
                                 "cards": []}
                tcl.set_output(error_payload)
                return json.dumps(error_payload, ensure_ascii=False)

            crm_filter = build_filter(
                module=module,
                search=search,
                filters=parsed_filters,
                date_from=date_from,
                date_to=date_to,
            )
            order = build_order(order_by)

            # print(f"Built CRM filter: {json.dumps(crm_filter, ensure_ascii=False)}")

            data: dict = {
                "limit": max(1, min(int(limit), 100)),
                "offset": 0,
                "filter": crm_filter,
                "include_field_names": False,
            }
            if order:
                data["order"] = order

            try:
                raw = await crm_client.execute_module_action(
                    module=module,
                    action="list",
                    data=data,
                )

                # print(f"Raw CRM response: {json.dumps(raw, ensure_ascii=False)}")
            except Exception as exc:
                error_payload = {"status": "error", "message": str(exc), "cards": []}
                tcl.set_output(error_payload)
                return json.dumps(error_payload, ensure_ascii=False)

            records = _extract_records(raw)

            # print(f"Extracted {len(records)} records from CRM response.")
            # print(f"First 10 records: {json.dumps(records[:10], ensure_ascii=False)}")

            cards: list[dict] = []
            module_lower = module.strip().lower()
            total = len(records)

            if module_lower == "meetings":
                cards = _meeting_cards(records, total)
            elif module_lower == "contacts":
                title = f"Kontakty ({total})"
                cards = _contact_cards(records, total, title, force_table=total > 4)
            else:
                cards = _generic_cards(records, total)

            # print(f"Generated {len(cards)} cards for module '{module}'.")
            # print(f"First 5 cards: {json.dumps(cards[:5], ensure_ascii=False)}")

            summary_lines = []
            for row in records[:10]:
                name = _record_name(row)
                extra = ""
                if module_lower == "contacts":
                    extra = f" — {_safe_text(row.get('account_name') or row.get('company'))}"
                elif module_lower in ("meetings", "calls"):
                    extra = f" — {_safe_text(row.get('date_start', ''))[:10]} {_safe_text(row.get('status', ''))}"
                summary_lines.append(f"• {name}{extra}".strip())

            result = {
                "status": "ok",
                "module": module,
                "total": total,
                "summary": "\n".join(summary_lines) if summary_lines else "Žádné záznamy.",
                "cards": cards,
            }
            tcl.set_output({"total": total, "module": module})
            return json.dumps(result, ensure_ascii=False)

    return [crm_action_tool, rag_search_tool, my_meetings_tool, crm_query_tool]
