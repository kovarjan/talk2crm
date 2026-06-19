# Copyright 2026 Jan Kovář
# Copyright 2026 Acmark s.r.o.
# SPDX-License-Identifier: Apache-2.0

"""Record extraction and normalization helpers for CRM/RAG payloads."""

from __future__ import annotations

import hashlib
import json
import re
from difflib import SequenceMatcher
from typing import Any

from app.presentation.cards import record_name
from app.utils.crm_id import CRM_ID_RE
from app.utils.text import normalize_text, safe_text

UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)
SCOPED_RECORD_ID_RE = re.compile(
    r"^(?P<module>[A-Za-z]+)[-:/](?P<id>(?:[0-9a-f]{32}|[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}))$",
    re.IGNORECASE,
)


def canonical_record_id(value: Any) -> str:
    text = safe_text(value)
    if not text:
        return ""
    if CRM_ID_RE.match(text):
        return text

    scoped = SCOPED_RECORD_ID_RE.match(text)
    if not scoped:
        return text

    candidate = safe_text(scoped.group("id"))
    return candidate if CRM_ID_RE.match(candidate) else text

def name_sim(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

def extract_name_value_scalar(raw: Any) -> Any:
    if isinstance(raw, dict):
        if "value" in raw:
            return raw.get("value")
        if "name" in raw and len(raw) == 1:
            return raw.get("name")
    return raw

def flatten_record(record: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(record, dict):
        return {}

    flattened = dict(record)
    name_value_list = flattened.get("name_value_list")
    if isinstance(name_value_list, dict):
        for key, value in name_value_list.items():
            if key not in flattened:
                flattened[key] = extract_name_value_scalar(value)

    attrs = flattened.get("attributes")
    if isinstance(attrs, dict):
        for key, value in attrs.items():
            if key not in flattened:
                flattened[key] = extract_name_value_scalar(value)

    return flattened

def extract_records(value: Any) -> list[dict[str, Any]]:
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
        record_id = safe_text(item.get("id"))
        if record_id and UUID_RE.match(record_id):
            return True

        # Fallback for some list/search responses that omit id but still hold entity row data.
        has_person_shape = bool(safe_text(item.get("first_name"))) and bool(safe_text(item.get("last_name")))
        has_company_shape = bool(safe_text(item.get("account_name") or item.get("company")))
        has_crm_dates = bool(safe_text(item.get("date_modified") or item.get("date_entered")))
        is_field_def_shape = "vname" in item and "type" in item and ("name" in item or "rname" in item)
        return (has_person_shape and (has_company_shape or has_crm_dates)) and not is_field_def_shape

    def walk(node: Any, parent_key: str = "") -> None:
        if isinstance(node, list):
            if parent_key in record_list_keys:
                for item in node:
                    if isinstance(item, dict) and is_record_candidate(item):
                        records.append(flatten_record(item))
            for item in node:
                walk(item, parent_key=parent_key)
            return

        if not isinstance(node, dict):
            return

        flattened = flatten_record(node)
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
        record_id = safe_text(item.get("id"))
        if record_id:
            if record_id in seen_ids:
                continue
            seen_ids.add(record_id)
        deduped.append(item)
    return deduped

def filter_valid_contact_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    valid: list[dict[str, Any]] = []
    for row in records:
        record_id = safe_text(row.get("id"))
        if not UUID_RE.match(record_id):
            continue

        name = safe_text(row.get("name"))
        first = safe_text(row.get("first_name"))
        last = safe_text(row.get("last_name"))
        email = safe_text(row.get("email1"))
        phone = safe_text(row.get("phone_mobile") or row.get("phone_work"))
        company = safe_text(row.get("account_name") or row.get("company"))
        if not any([name, first, last, email, phone, company]):
            continue
        valid.append(row)
    return valid

def best_record_match(records: list[dict[str, Any]], search: str) -> dict[str, Any] | None:
    if not records:
        return None
    search_norm = normalize_text(search)
    if not search_norm:
        return None
    best: tuple[int, dict[str, Any]] | None = None
    for record in records:
        candidate = normalize_text(record_name(record))
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

def extract_rag_records(
    results: list[dict[str, Any]],
    *,
    module_hint: str | None = None,
) -> list[dict[str, Any]]:
    wanted = safe_text(module_hint).lower()
    rows: list[dict[str, Any]] = []

    for item in results:
        if not isinstance(item, dict):
            continue
        payload = item.get("payload")
        if not isinstance(payload, dict):
            continue
        payload_module = safe_text(payload.get("module")).lower()
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
        row_id = safe_text(row.get("id"))
        if not row_id:
            no_id.append(row)
            continue
        prev = deduped.get(row_id)
        if prev is None or float(row.get("_rag_score", 0.0)) > float(prev.get("_rag_score", 0.0)):
            deduped[row_id] = row
    return list(deduped.values()) + no_id

def record_primary_name(record: dict[str, Any]) -> str:
    if not isinstance(record, dict):
        return ""
    direct = safe_text(record.get("name") or record.get("NAME"))
    if direct:
        return direct
    alt = safe_text(record.get("account_name") or record.get("company") or record.get("ACCOUNT_NAME"))
    if alt:
        return alt
    first = safe_text(record.get("first_name") or record.get("FIRST_NAME"))
    last = safe_text(record.get("last_name") or record.get("LAST_NAME"))
    full = f"{first} {last}".strip()
    return full

def compact_rag_payload(payload: dict[str, Any]) -> dict[str, Any]:
    compact = dict(payload)
    text = compact.get("text")
    record = compact.get("record")
    if isinstance(text, dict) and isinstance(record, dict) and text == record:
        # Avoid returning the same record twice in payload under both "text" and "record".
        compact.pop("text", None)
    return compact

def rag_result_identity(item: dict[str, Any]) -> str:
    payload = item.get("payload")
    if not isinstance(payload, dict):
        return ""
    module_name = safe_text(payload.get("module")).lower()
    record = payload.get("record") if isinstance(payload.get("record"), dict) else {}
    text = payload.get("text") if isinstance(payload.get("text"), dict) else {}
    record_id = canonical_record_id(
        payload.get("record_id")
        or record.get("id")
        or text.get("id")
    )
    if record_id:
        return f"id:{module_name}:{record_id.lower()}"

    merged: dict[str, Any] = {}
    if isinstance(text, dict):
        merged.update(text)
    if isinstance(record, dict):
        merged.update(record)

    primary_name = normalize_text(record_primary_name(merged))
    company = normalize_text(safe_text(merged.get("account_name") or merged.get("company")))
    city = normalize_text(
        safe_text(merged.get("primary_address_city") or merged.get("billing_address_city"))
    )
    street = normalize_text(
        safe_text(merged.get("primary_address_street") or merged.get("billing_address_street"))
    )
    phone = safe_text(
        merged.get("phone_work")
        or merged.get("phone_mobile")
        or merged.get("phone_office")
        or merged.get("phone_home")
    )
    email = normalize_text(safe_text(merged.get("email1") or merged.get("email")))
    fingerprint_parts = [module_name, primary_name, company, city, street, phone, email]
    if any(fingerprint_parts[1:]):
        return "shape:" + "|".join(fingerprint_parts)

    try:
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    except Exception:
        raw = safe_text(payload)
    return "raw:" + hashlib.sha1(raw.encode("utf-8")).hexdigest()

def rag_result_score(item: dict[str, Any]) -> float:
    try:
        return float(item.get("score") or 0.0)
    except Exception:
        return 0.0

def dedupe_rag_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    index_by_identity: dict[str, int] = {}
    for item in results:
        if not isinstance(item, dict):
            continue
        candidate = dict(item)
        payload = candidate.get("payload")
        if isinstance(payload, dict):
            candidate["payload"] = compact_rag_payload(payload)
        identity = rag_result_identity(candidate)
        if not identity:
            try:
                raw = json.dumps(candidate, ensure_ascii=False, sort_keys=True, default=str)
            except Exception:
                raw = safe_text(candidate)
            identity = "raw_item:" + hashlib.sha1(raw.encode("utf-8")).hexdigest()
        if identity not in index_by_identity:
            index_by_identity[identity] = len(deduped)
            deduped.append(candidate)
            continue

        existing_index = index_by_identity[identity]
        if rag_result_score(candidate) > rag_result_score(deduped[existing_index]):
            deduped[existing_index] = candidate
    return deduped

def score_text_match(search: str, candidate: str) -> int:
    s = normalize_text(search)
    c = normalize_text(candidate)
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

def filter_records_by_company(
    records: list[dict[str, Any]],
    company_name: str,
    *,
    fallback_to_original: bool = True,
) -> list[dict[str, Any]]:
    normalized_company = normalize_text(company_name)
    if not normalized_company:
        return records

    filtered: list[dict[str, Any]] = []
    for record in records:
        account = normalize_text(safe_text(record.get("account_name") or record.get("company")))
        if normalized_company in account or account in normalized_company:
            filtered.append(record)
    if filtered:
        return filtered
    return records if fallback_to_original else []

def dedupe_and_limit(records: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for record in records:
        record_id = safe_text(record.get("id"))
        name = normalize_text(record_name(record))
        fingerprint = f"{record_id}|{name}"
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        deduped.append(record)
        if len(deduped) >= limit:
            break
    return deduped

def merge_records_by_id(*record_sets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen: set[str] = set()
    for rows in record_sets:
        for row in rows:
            rec_id = safe_text(row.get("id"))
            if not rec_id or rec_id in seen:
                continue
            seen.add(rec_id)
            merged.append(row)
    return merged
