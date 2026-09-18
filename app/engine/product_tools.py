# Copyright 2026 Jan Kovář
# Copyright 2026 Acmark s.r.o.
# SPDX-License-Identifier: Apache-2.0
"""Product catalog lookup (ProductTemplates) for the ``products`` capability.

Live catalog lookup followed by semantic search in the tenant's Qdrant collection,
re-ranked by the shared fuzzy scorer; exact codes and names win. Prices are catalog prices only —
the FE applies price-list pricing when a row is inserted into a quote.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from app.engine.filter_builder import FilterSpec, build_filter
from app.presentation.cards import table_card
from app.utils.fuzzy import score_lexical_fuzzy
from app.utils.text import normalize_text

logger = logging.getLogger(__name__)

MODULE = "ProductTemplates"
NOTE_CATALOG = "Ceny jsou katalogové; cena pro nabídku se dopočítá podle ceníků při vložení řádku."


class ProductLookupArgs(BaseModel):
    query: str = Field(description="Název, část názvu nebo kód (mft_part_num) produktu")
    limit: int = Field(default=5, ge=1, le=20)


def _num(value: Any) -> float | None:
    if value in (None, "", False):
        return None
    try:
        return float(str(value).replace(",", "."))
    except ValueError:
        return None


def _to_result(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": str(record.get("id") or ""),
        "name": str(record.get("name") or ""),
        "mft_part_num": str(record.get("mft_part_num") or ""),
        "manufacturer": str(record.get("manufacturer_name") or ""),
        "category": str(record.get("category_name") or ""),
        "list_price": _num(record.get("list_price")),
        "cost_price": _num(record.get("cost_price")),
        "currency_id": str(record.get("currency_id") or "-99"),
        "price_source": "catalog",
    }


def rank_products(query: str, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    q = normalize_text(query)
    scored: list[tuple[float, dict[str, Any]]] = []
    for record in records:
        part = normalize_text(str(record.get("mft_part_num") or ""))
        name = normalize_text(str(record.get("name") or ""))
        if part and part == q:
            score = 10.0
        elif name and name == q:
            score = 9.0
        else:
            score = max(score_lexical_fuzzy(q, name), score_lexical_fuzzy(q, part) if part else 0.0)
        scored.append((score, record))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [record for _, record in scored]


def _cards(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not results:
        return []
    keys = ("name", "mft_part_num", "list_price", "manufacturer")
    return [table_card(
        title="Produkty",
        tag="ProductTemplates",
        columns=[{"key": "name", "label": "Název"}, {"key": "mft_part_num", "label": "Kód"},
                 {"key": "list_price", "label": "Cena"}, {"key": "manufacturer", "label": "Výrobce"}],
        rows=[{"id": r["id"], "module": MODULE, "cells": {k: r[k] for k in keys}} for r in results],
    )]


def _records_from_hits(hits: list[Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for hit in hits or []:
        payload = hit.get("payload") if isinstance(hit, dict) else None
        if not isinstance(payload, dict):
            continue
        record = payload.get("record") if isinstance(payload.get("record"), dict) else payload
        if record:
            records.append(record)
    return records


def build_product_tools(*, tenant_id: str, user_id: str, crm_client: Any, rag_service: Any | None) -> list:
    @tool("product_lookup_tool", args_schema=ProductLookupArgs)
    async def product_lookup_tool(query: str, limit: int = 5) -> str:
        """Najde produkty v katalogu (ProductTemplates) podle názvu nebo kódu a vrátí jejich id a katalogové ceny."""
        records: list[dict[str, Any]] = []
        status = "ok"
        fetch = max(limit * 4, 20)
        # Vector hits may be non-empty even when the exact catalog product was
        # never indexed. Always query the authoritative catalog before using RAG.
        query = query.strip()
        if not query:
            return json.dumps({"status": "ok", "results": [], "cards": [], "note": NOTE_CATALOG})
        try:
            exact_filter = {
                "operator": "or",
                "operands": [build_filter(module=MODULE, filters=[FilterSpec(field=field, op="eq", value=query)])
                             for field in ("name", "mft_part_num")],
            }
            payload = await crm_client.execute_module_action(
                module=MODULE, action="list",
                data={"filter": exact_filter, "limit": fetch, "offset": 0},
            )
            records = [r for r in (payload.get("records") or []) if isinstance(r, dict)]
            exact = [r for r in records if normalize_text(query) in (
                normalize_text(str(r.get("name") or "")), normalize_text(str(r.get("mft_part_num") or "")))]
            if exact:
                records = exact
            else:
                # AND the name tokens so punctuation/spacing differences do not
                # hide e.g. "Kompresor KAB, 12V" behind an unrelated first page.
                tokens = re.findall(r"[^\W_]+", query, flags=re.UNICODE)
                name_filter = build_filter(module=MODULE, filters=[
                    FilterSpec(field="name", op="cont", value=token) for token in tokens
                ])
                code_filter = build_filter(module=MODULE, filters=[FilterSpec(field="mft_part_num", op="cont", value=query)])
                payload = await crm_client.execute_module_action(
                    module=MODULE, action="list",
                    data={"filter": {"operator": "or", "operands": [name_filter, code_filter]} if tokens else code_filter,
                          "limit": fetch, "offset": 0},
                )
                records = [r for r in (payload.get("records") or []) if isinstance(r, dict)]
        except Exception as exc:  # noqa: BLE001
            logger.warning("product lookup: crm search failed tenant=%s error=%s", tenant_id, exc)
            status = "degraded"
        has_exact = any(normalize_text(query) in (
            normalize_text(str(r.get("name") or "")), normalize_text(str(r.get("mft_part_num") or ""))) for r in records)
        if rag_service is not None and not has_exact:
            try:
                hits = await rag_service.search(tenant_id=tenant_id, query=query, limit=fetch, modules=[MODULE])
                # Fresh CRM values take precedence over cached index values.
                seen = {r.get("id") for r in records}
                for record in _records_from_hits(hits):
                    if record.get("id") not in seen:
                        records.append(record)
                        seen.add(record.get("id"))
            except Exception as exc:  # noqa: BLE001 - degrade, never fail the turn
                logger.warning("product lookup: qdrant unavailable tenant=%s error=%s", tenant_id, exc)
                status = "degraded"
        ranked = rank_products(query, records)[:limit]
        results = [_to_result(r) for r in ranked if r.get("id")]
        return json.dumps({"status": status, "results": results, "cards": _cards(results), "note": NOTE_CATALOG}, ensure_ascii=False)

    return [product_lookup_tool]
