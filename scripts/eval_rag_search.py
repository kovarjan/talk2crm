"""RAG search accuracy benchmark — semantic-only vs hybrid.

Usage (from project root, Docker stack running):
    source .venv/bin/activate
    python scripts/eval_rag_search.py

Outputs:
    eval_results_raw.csv      — one row per (query × mode)
    eval_results_summary.csv  — Hit Rate@1 per category, semantic vs hybrid
"""
from __future__ import annotations

import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Allow importing the app package from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.config import get_settings
from app.engine.rag import TenantRAGService

TENANT_ID = "ai-local"
LIMIT = 5  # k=1, k=3, k=5 are all evaluated from the same top-5 list

RAW_CSV = Path("eval_results_raw.csv")
SUMMARY_CSV = Path("eval_results_summary.csv")


@dataclass
class TestCase:
    category: str           # thesis category label (Czech)
    record_label: str       # short human label for the target record
    query: str              # query string sent to the search
    expected_ids: list[str] # any of these record_ids counts as a hit (branches/duplicates)


# 25 cases: 5 records × 5 query categories
# Casing rules:
#   Přesný název  → exact CRM casing
#   all others    → lowercase / sentence-case (realistic user input)
_CEMAT_IDS = [
    "151cd173-c17e-3622-9fe1-5f2ac1d7ed6b",  # ČEMAT trading, spol. s r.o.
    "73c979c2-0efd-eb5c-551a-62847462aebc",  # ČEMAT, s.r.o. (branch 1)
    "c92c789a-ad03-4858-8a1e-fa16c6001787",  # ČEMAT, s.r.o. (branch 2)
]

_TESCAN_IDS = [
    "ed4e5c66-3c98-8b7f-7961-524d6749513a",  # TESCAN GROUP, a.s.
    "4ab27509-cb19-487d-3b8c-64e3303521c9",  # TESCAN Medical, s.r.o.
]

TEST_CASES: list[TestCase] = [
    # --- Přesný název (exact CRM name) ---
    TestCase("Přesný název",   "ČEMAT",     "ČEMAT trading, spol. s r.o.",  _CEMAT_IDS),
    TestCase("Přesný název",   "TESCAN",    "TESCAN GROUP, a.s.",           _TESCAN_IDS),
    TestCase("Přesný název",   "Sekyrová",  "Ilona Sekyrová",               ["753e8ad7-e827-cae2-39fd-55bb17146044"]),
    TestCase("Přesný název",   "Předešlý",  "Libor Předešlý",               ["8ae00617-a02d-19e1-2edd-6026b14e2046"]),
    TestCase("Přesný název",   "Pětvaldský","Karel Pětvaldský",             ["f134dbd7-91ea-c49d-66c9-5f3e69da202c"]),

    # --- Bez diakritiky (no diacritics, lowercase) ---
    TestCase("Bez diakritiky", "ČEMAT",     "cemat trading",                _CEMAT_IDS),
    TestCase("Bez diakritiky", "TESCAN",    "tescan group a.s.",            _TESCAN_IDS),
    TestCase("Bez diakritiky", "Sekyrová",  "ilona sekyrova",               ["753e8ad7-e827-cae2-39fd-55bb17146044"]),
    TestCase("Bez diakritiky", "Předešlý",  "libor predesly",               ["8ae00617-a02d-19e1-2edd-6026b14e2046"]),
    TestCase("Bez diakritiky", "Pětvaldský","karel petvaldsky",             ["f134dbd7-91ea-c49d-66c9-5f3e69da202c"]),

    # --- Skloňovaný tvar (inflected, lowercase / sentence-case) ---
    TestCase("Skloňovaný tvar", "ČEMAT",    "v čematu",                     _CEMAT_IDS),
    TestCase("Skloňovaný tvar", "TESCAN",   "do tescanu",                   _TESCAN_IDS),
    TestCase("Skloňovaný tvar", "Sekyrová", "ilony sekyrové",               ["753e8ad7-e827-cae2-39fd-55bb17146044"]),
    TestCase("Skloňovaný tvar", "Předešlý", "libora předešlého",            ["8ae00617-a02d-19e1-2edd-6026b14e2046"]),
    TestCase("Skloňovaný tvar", "Pětvaldský","karla pětvaldského",          ["f134dbd7-91ea-c49d-66c9-5f3e69da202c"]),

    # --- Překlep / STT chyba (typo / speech-to-text error, lowercase) ---
    TestCase("Překlep / STT chyba", "ČEMAT",    "šemat trejding",           _CEMAT_IDS),
    TestCase("Překlep / STT chyba", "TESCAN",   "teskan group",             _TESCAN_IDS),
    TestCase("Překlep / STT chyba", "Sekyrová", "ilona sekirova",           ["753e8ad7-e827-cae2-39fd-55bb17146044"]),
    TestCase("Překlep / STT chyba", "Předešlý", "libor přeďešlý",          ["8ae00617-a02d-19e1-2edd-6026b14e2046"]),
    TestCase("Překlep / STT chyba", "Pětvaldský","karel petvaldski",        ["f134dbd7-91ea-c49d-66c9-5f3e69da202c"]),

    # --- Neúplný dotaz (partial name, lowercase) ---
    TestCase("Neúplný dotaz", "ČEMAT",     "čemat",                         _CEMAT_IDS),
    TestCase("Neúplný dotaz", "TESCAN",    "tescan",                        _TESCAN_IDS),
    TestCase("Neúplný dotaz", "Sekyrová",  "sekyrová",                      ["753e8ad7-e827-cae2-39fd-55bb17146044"]),
    TestCase("Neúplný dotaz", "Předešlý",  "předešlý",                      ["8ae00617-a02d-19e1-2edd-6026b14e2046"]),
    TestCase("Neúplný dotaz", "Pětvaldský","pětvaldský",                    ["f134dbd7-91ea-c49d-66c9-5f3e69da202c"]),
]


def _search_hybrid(
    rag_service: TenantRAGService,
    tenant_id: str,
    query: str,
    limit: int,
) -> list[dict[str, Any]]:
    """Hybrid search: text-filter pass (score 1.0) merged with vector similarity.
    This is the exact path TenantRAGService.search() uses in production."""
    return rag_service.search(tenant_id=tenant_id, query=query, limit=limit)


def _search_semantic(
    rag_service: TenantRAGService,
    tenant_id: str,
    query: str,
    limit: int,
) -> list[dict[str, Any]]:
    """Semantic-only search: pure vector similarity, no text-filter pre-pass.
    Uses the same client and embedder as the hybrid path for a fair comparison."""
    collection = rag_service._tenant_collection(tenant_id)
    query_vector = rag_service.embedder.embed(query)
    try:
        if hasattr(rag_service.client, "search"):
            points = rag_service.client.search(
                collection_name=collection,
                query_vector=query_vector,
                limit=limit,
                with_payload=True,
            )
        else:
            points = rag_service.client.query_points(
                collection_name=collection,
                query=query_vector,
                limit=limit,
                with_payload=True,
            ).points
        return [{"score": p.score, "payload": p.payload} for p in points]
    except Exception as exc:
        print(f"  [semantic error] {exc}")
        return []


def _hit_at_k(results: list[dict[str, Any]], expected_ids: list[str], k: int) -> int:
    """Return 1 if any expected_id appears in the top-k results, else 0."""
    for r in results[:k]:
        if (r.get("payload") or {}).get("record_id") in expected_ids:
            return 1
    return 0


def _top1_info(results: list[dict[str, Any]]) -> tuple[str, str, float]:
    """Return (record_id, name, score) for the first result, or empty strings."""
    if not results:
        return "", "", 0.0
    payload = results[0].get("payload") or {}
    record = payload.get("record") or {}
    record_id = payload.get("record_id", "")
    name = record.get("name", "")
    score = float(results[0].get("score", 0.0))
    return record_id, name, score


CATEGORIES_ORDER = [
    "Přesný název",
    "Bez diakritiky",
    "Skloňovaný tvar",
    "Překlep / STT chyba",
    "Neúplný dotaz",
]


def main() -> None:
    from collections import defaultdict

    print("RAG search accuracy benchmark")
    print(f"Tenant: {TENANT_ID}  |  Cases: {len(TEST_CASES)}  |  Modes: semantic, hybrid")
    print()

    print("Initialising TenantRAGService (connects to Qdrant)...")
    rag_service = TenantRAGService()
    print(f"  Embedder: {type(rag_service.embedder).__name__}")
    print()

    @dataclass
    class Row:
        category: str
        record_label: str
        query: str
        mode: str
        hit1: int
        hit3: int
        hit5: int
        top1_id: str
        top1_name: str
        top1_score: float

    rows: list[Row] = []

    for tc in TEST_CASES:
        for mode, search_fn in [
            ("semantic", _search_semantic),
            ("hybrid",   _search_hybrid),
        ]:
            results = search_fn(rag_service, TENANT_ID, tc.query, LIMIT)
            h1 = _hit_at_k(results, tc.expected_ids, 1)
            h3 = _hit_at_k(results, tc.expected_ids, 3)
            h5 = _hit_at_k(results, tc.expected_ids, 5)
            top_id, top_name, top_score = _top1_info(results)

            status = "✓" if h1 else ("~" if h3 else "✗")
            print(f"[{mode:8s}] {status} {tc.category:22s} | {tc.record_label:12s} | {tc.query!r}")
            if not h1:
                print(f"           → top1: {top_name!r} ({top_id[:8]}…) score={top_score:.3f}")

            rows.append(Row(
                category=tc.category,
                record_label=tc.record_label,
                query=tc.query,
                mode=mode,
                hit1=h1,
                hit3=h3,
                hit5=h5,
                top1_id=top_id,
                top1_name=top_name,
                top1_score=top_score,
            ))

    # Write raw CSV
    with RAW_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "category", "record_label", "query", "mode",
            "hit@1", "hit@3", "hit@5",
            "top1_record_id", "top1_name", "top1_score",
        ])
        for r in rows:
            writer.writerow([
                r.category, r.record_label, r.query, r.mode,
                r.hit1, r.hit3, r.hit5,
                r.top1_id, r.top1_name, f"{r.top1_score:.4f}",
            ])
    print(f"\nRaw results → {RAW_CSV}")

    # Compute summary: Hit Rate@1 per category per mode
    counts: dict[tuple[str, str], list[int]] = defaultdict(list)
    for r in rows:
        counts[(r.category, r.mode)].append(r.hit1)

    sem_totals: list[float] = []
    hyb_totals: list[float] = []

    with SUMMARY_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["category", "semantic_hit@1", "hybrid_hit@1"])

        for cat in CATEGORIES_ORDER:
            sem_hits = counts.get((cat, "semantic"), [])
            hyb_hits = counts.get((cat, "hybrid"), [])
            sem_rate = sum(sem_hits) / len(sem_hits) if sem_hits else 0.0
            hyb_rate = sum(hyb_hits) / len(hyb_hits) if hyb_hits else 0.0
            sem_totals.append(sem_rate)
            hyb_totals.append(hyb_rate)
            writer.writerow([cat, f"{sem_rate:.2f}", f"{hyb_rate:.2f}"])

        avg_sem = sum(sem_totals) / len(sem_totals) if sem_totals else 0.0
        avg_hyb = sum(hyb_totals) / len(hyb_totals) if hyb_totals else 0.0
        writer.writerow(["Průměr", f"{avg_sem:.2f}", f"{avg_hyb:.2f}"])

    print(f"Summary      → {SUMMARY_CSV}")

    # Print summary table to stdout
    print()
    print(f"{'Kategorie':<25} {'Sémantické':>12} {'Hybridní':>10}")
    print("-" * 50)
    for cat, sem_rate, hyb_rate in zip(CATEGORIES_ORDER, sem_totals, hyb_totals):
        print(f"{cat:<25} {sem_rate:>12.0%} {hyb_rate:>10.0%}")
    print("-" * 50)
    print(f"{'Průměr':<25} {avg_sem:>12.0%} {avg_hyb:>10.0%}")


if __name__ == "__main__":
    main()
