# RAG Search Accuracy Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce `scripts/eval_rag_search.py` — a standalone benchmark script that compares semantic-only vs hybrid RAG search across 5 Czech CRM query categories and writes two CSVs mapping directly to thesis Table 5.

**Architecture:** The script imports `TenantRAGService` from the project for the hybrid path (exact production code), and calls `qdrant_client.search()` directly (skipping the text-filter pass) for the semantic-only baseline. Both modes share the same embedder instance extracted from `TenantRAGService`. 25 hardcoded test cases (5 records × 5 categories) are evaluated for Hit Rate@1, @3, @5.

**Tech Stack:** Python 3.11, qdrant-client, TenantRAGService (app/engine/rag.py), pydantic-settings (app/core/config.py), csv stdlib

---

## File Structure

| Action | Path | Responsibility |
|---|---|---|
| Create | `scripts/eval_rag_search.py` | Full benchmark script — test cases, search modes, metric calc, CSV output |

No other files are created or modified.

---

### Task 1: Scaffold the script with config, test cases, and CSV skeleton

**Files:**
- Create: `scripts/eval_rag_search.py`

- [ ] **Step 1: Create the file with imports, test cases, and a no-op main**

`scripts/eval_rag_search.py`:

```python
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
    category: str        # thesis category label (Czech)
    record_label: str    # short human label for the target record
    query: str           # query string sent to the search
    expected_id: str     # record_id that must appear in results


# 25 cases: 5 records × 5 query categories
# Casing rules:
#   Přesný název  → exact CRM casing
#   all others    → lowercase / sentence-case (realistic user input)
TEST_CASES: list[TestCase] = [
    # --- Přesný název (exact CRM name) ---
    TestCase("Přesný název",   "ČEMAT",      "ČEMAT trading, spol. s r.o.",        "151cd173-c17e-3622-9fe1-5f2ac1d7ed6b"),
    TestCase("Přesný název",   "ACMARK",     "ACMARK s.r.o.",                       "82b6c408-be0f-a4c6-fadc-66446fb0c14d"),
    TestCase("Přesný název",   "Bednář",     "Patrik Bednář",                       "8d1d638f-a107-13e0-f7f8-60479d8f9137"),
    TestCase("Přesný název",   "Předešlý",   "Libor Předešlý",                      "8ae00617-a02d-19e1-2edd-6026b14e2046"),
    TestCase("Přesný název",   "Pětvaldský", "Karel Pětvaldský",                    "f134dbd7-91ea-c49d-66c9-5f3e69da202c"),

    # --- Bez diakritiky (no diacritics, lowercase) ---
    TestCase("Bez diakritiky", "ČEMAT",      "cemat trading",                       "151cd173-c17e-3622-9fe1-5f2ac1d7ed6b"),
    TestCase("Bez diakritiky", "ACMARK",     "acmark s.r.o.",                       "82b6c408-be0f-a4c6-fadc-66446fb0c14d"),
    TestCase("Bez diakritiky", "Bednář",     "patrik bednar",                       "8d1d638f-a107-13e0-f7f8-60479d8f9137"),
    TestCase("Bez diakritiky", "Předešlý",   "libor predesly",                      "8ae00617-a02d-19e1-2edd-6026b14e2046"),
    TestCase("Bez diakritiky", "Pětvaldský", "karel petvaldsky",                    "f134dbd7-91ea-c49d-66c9-5f3e69da202c"),

    # --- Skloňovaný tvar (inflected, lowercase / sentence-case) ---
    TestCase("Skloňovaný tvar", "ČEMAT",      "v čematu",                           "151cd173-c17e-3622-9fe1-5f2ac1d7ed6b"),
    TestCase("Skloňovaný tvar", "ACMARK",     "do acmarku",                         "82b6c408-be0f-a4c6-fadc-66446fb0c14d"),
    TestCase("Skloňovaný tvar", "Bednář",     "Patrika Bednáře",                    "8d1d638f-a107-13e0-f7f8-60479d8f9137"),
    TestCase("Skloňovaný tvar", "Předešlý",   "libora předešlého",                  "8ae00617-a02d-19e1-2edd-6026b14e2046"),
    TestCase("Skloňovaný tvar", "Pětvaldský", "karla pětvaldského",                 "f134dbd7-91ea-c49d-66c9-5f3e69da202c"),

    # --- Překlep / STT chyba (typo / speech-to-text error, lowercase) ---
    TestCase("Překlep / STT chyba", "ČEMAT",      "šemat trejding",                "151cd173-c17e-3622-9fe1-5f2ac1d7ed6b"),
    TestCase("Překlep / STT chyba", "ACMARK",     "axmark s.r.o.",                 "82b6c408-be0f-a4c6-fadc-66446fb0c14d"),
    TestCase("Překlep / STT chyba", "Bednář",     "patrik bednat",                 "8d1d638f-a107-13e0-f7f8-60479d8f9137"),
    TestCase("Překlep / STT chyba", "Předešlý",   "libor přeďešlý",               "8ae00617-a02d-19e1-2edd-6026b14e2046"),
    TestCase("Překlep / STT chyba", "Pětvaldský", "karel petvaldski",              "f134dbd7-91ea-c49d-66c9-5f3e69da202c"),

    # --- Neúplný dotaz (partial name, lowercase) ---
    TestCase("Neúplný dotaz", "ČEMAT",      "čemat",                               "151cd173-c17e-3622-9fe1-5f2ac1d7ed6b"),
    TestCase("Neúplný dotaz", "ACMARK",     "acmark",                              "82b6c408-be0f-a4c6-fadc-66446fb0c14d"),
    TestCase("Neúplný dotaz", "Bednář",     "bednář",                              "8d1d638f-a107-13e0-f7f8-60479d8f9137"),
    TestCase("Neúplný dotaz", "Předešlý",   "předešlý",                            "8ae00617-a02d-19e1-2edd-6026b14e2046"),
    TestCase("Neúplný dotaz", "Pětvaldský", "pětvaldský",                          "f134dbd7-91ea-c49d-66c9-5f3e69da202c"),
]


def main() -> None:
    print("RAG search accuracy benchmark")
    print(f"Tenant: {TENANT_ID}  |  Cases: {len(TEST_CASES)}  |  Modes: semantic, hybrid")
    print()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify the script is importable and runs**

```bash
cd /path/to/project && source .venv/bin/activate
python scripts/eval_rag_search.py
```

Expected output:
```
RAG search accuracy benchmark
Tenant: ai-local  |  Cases: 25  |  Modes: semantic, hybrid
```

---

### Task 2: Add the two search functions

**Files:**
- Modify: `scripts/eval_rag_search.py`

- [ ] **Step 1: Add `_search_hybrid` and `_search_semantic` above `main()`**

Insert these two functions into `scripts/eval_rag_search.py` before the `main()` function:

```python
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


def _hit_at_k(results: list[dict[str, Any]], expected_id: str, k: int) -> int:
    """Return 1 if expected_id appears in the top-k results, else 0."""
    for r in results[:k]:
        if (r.get("payload") or {}).get("record_id") == expected_id:
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
```

- [ ] **Step 2: Verify the script still runs**

```bash
python scripts/eval_rag_search.py
```

Expected: same output as before — no import errors.

---

### Task 3: Wire up the evaluation loop and CSV output

**Files:**
- Modify: `scripts/eval_rag_search.py`

- [ ] **Step 1: Replace the `main()` stub with the full evaluation loop**

Replace the existing `main()` function:

```python
CATEGORIES_ORDER = [
    "Přesný název",
    "Bez diakritiky",
    "Skloňovaný tvar",
    "Překlep / STT chyba",
    "Neúplný dotaz",
]

def main() -> None:
    print("RAG search accuracy benchmark")
    print(f"Tenant: {TENANT_ID}  |  Cases: {len(TEST_CASES)}  |  Modes: semantic, hybrid")
    print()

    print("Initialising TenantRAGService (connects to Qdrant)...")
    rag_service = TenantRAGService()
    print(f"  Embedder: {type(rag_service.embedder).__name__}")
    print()

    # ------------------------------------------------------------------ #
    #  Run all test cases for both modes                                   #
    # ------------------------------------------------------------------ #
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
            h1 = _hit_at_k(results, tc.expected_id, 1)
            h3 = _hit_at_k(results, tc.expected_id, 3)
            h5 = _hit_at_k(results, tc.expected_id, 5)
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

    # ------------------------------------------------------------------ #
    #  Write raw CSV                                                        #
    # ------------------------------------------------------------------ #
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

    # ------------------------------------------------------------------ #
    #  Compute summary: Hit Rate@1 per category per mode                   #
    # ------------------------------------------------------------------ #
    from collections import defaultdict
    counts: dict[tuple[str, str], list[int]] = defaultdict(list)
    for r in rows:
        counts[(r.category, r.mode)].append(r.hit1)

    with SUMMARY_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["category", "semantic_hit@1", "hybrid_hit@1"])

        sem_totals: list[float] = []
        hyb_totals: list[float] = []

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

    # ------------------------------------------------------------------ #
    #  Print summary table to stdout                                        #
    # ------------------------------------------------------------------ #
    print()
    print(f"{'Kategorie':<25} {'Sémantické':>12} {'Hybridní':>10}")
    print("-" * 50)
    for cat in CATEGORIES_ORDER:
        sem_hits = counts.get((cat, "semantic"), [])
        hyb_hits = counts.get((cat, "hybrid"), [])
        sem_rate = sum(sem_hits) / len(sem_hits) if sem_hits else 0.0
        hyb_rate = sum(hyb_hits) / len(hyb_hits) if hyb_hits else 0.0
        print(f"{cat:<25} {sem_rate:>12.0%} {hyb_rate:>10.0%}")
    print("-" * 50)
    print(f"{'Průměr':<25} {avg_sem:>12.0%} {avg_hyb:>10.0%}")
```

- [ ] **Step 2: Add missing import at the top of the file**

The `defaultdict` import is inside `main()` already (added inline above). No extra top-level import needed.

- [ ] **Step 3: Run the full benchmark against the live Docker stack**

```bash
python scripts/eval_rag_search.py
```

Expected console output shape (values depend on real embeddings):
```
RAG search accuracy benchmark
Tenant: ai-local  |  Cases: 25  |  Modes: semantic, hybrid

Initialising TenantRAGService (connects to Qdrant)...
  Embedder: OllamaEmbedder

[semantic ] ✓ Přesný název          | ČEMAT        | 'ČEMAT trading, spol. s r.o.'
[hybrid   ] ✓ Přesný název          | ČEMAT        | 'ČEMAT trading, spol. s r.o.'
...

Raw results → eval_results_raw.csv
Summary      → eval_results_summary.csv

Kategorie                 Sémantické    Hybridní
--------------------------------------------------
Přesný název                     80%       100%
Bez diakritiky                   40%        60%
...
Průměr                           55%        75%
```

- [ ] **Step 4: Verify both CSV files exist and have expected shape**

```bash
head -3 eval_results_raw.csv
# Expected: header row + 2 data rows (one per mode for first test case)

wc -l eval_results_raw.csv
# Expected: 51 (1 header + 50 data rows = 25 cases × 2 modes)

wc -l eval_results_summary.csv
# Expected: 7 (1 header + 5 categories + 1 Průměr row)
```

---

## Self-Review

**Spec coverage:**
- ✓ Standalone script at `scripts/eval_rag_search.py`
- ✓ Semantic mode: Qdrant `.search()` directly, no text filter
- ✓ Hybrid mode: `TenantRAGService.search()` — exact production path
- ✓ Same embedder for both modes (fair comparison)
- ✓ 25 test cases: 5 records × 5 categories
- ✓ Casing rule: exact CRM for "Přesný název", lowercase for all others
- ✓ Hit Rate@1, @3, @5 computed
- ✓ `eval_results_raw.csv` — one row per query × mode
- ✓ `eval_results_summary.csv` — maps to thesis Table 5, includes Průměr
- ✓ Run command documented
- ✓ OllamaEmbedder / HashEmbedder fallback handled transparently

**Placeholder scan:** None found. All code blocks are complete.

**Type consistency:**
- `_search_hybrid` and `_search_semantic` both return `list[dict[str, Any]]` — consumed identically in the loop ✓
- `_hit_at_k` takes `list[dict[str, Any]]` and `int` — called with the same type ✓
- `_top1_info` returns `(str, str, float)` — destructured as `top_id, top_name, top_score` ✓
- `Row.top1_score` is `float`, formatted as `f"{r.top1_score:.4f}"` ✓
- `counts` keys are `(category: str, mode: str)` — looked up with `(cat, "semantic")` and `(cat, "hybrid")` ✓
