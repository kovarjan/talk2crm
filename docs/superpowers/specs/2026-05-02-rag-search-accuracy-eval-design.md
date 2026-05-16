# RAG Search Accuracy Evaluation Design

**Date:** 2026-05-02  
**Author:** Jan Kovář  
**Purpose:** Bachelor's thesis — compare semantic-only vs hybrid RAG search across Czech CRM query categories

---

## Goal

Produce a reproducible benchmark that fills Table 5 in the thesis: Hit Rate@k per query category for semantic-only search vs the hybrid approach actually used by the LLM agent tool `rag_search_tool`.

---

## Scope

Single standalone script: `scripts/eval_rag_search.py`  
Tenant: `ai-local`  
Data: real Qdrant vectors already ingested  
Output: two CSV files (raw + summary)

---

## Search Modes

### Semantic only (baseline)
Calls Qdrant `.search()` directly with the embedded query vector — no text filter pre-pass.  
Implements the pure vector similarity path in isolation.

### Hybrid (production)
Calls `TenantRAGService.search()` as-is — combines Qdrant full-text index (MatchText, score 1.0) with vector similarity, merged and deduplicated.  
This is the exact path used by `rag_search_tool` inside the LangChain agent.

Both modes use the same embedder (`OllamaEmbedder` when `rag_embedding_base_url` is configured, `HashEmbedder` fallback).

---

## Test Cases

25 cases total: 5 records × 5 query categories.

### Records under test

| Module | Name | record_id |
|---|---|---|
| Accounts | ČEMAT trading, spol. s r.o. | `151cd173-c17e-3622-9fe1-5f2ac1d7ed6b` |
| Accounts | ACMARK s.r.o. | `82b6c408-be0f-a4c6-fadc-66446fb0c14d` |
| Contacts | Patrik Bednář | `8d1d638f-a107-13e0-f7f8-60479d8f9137` |
| Contacts | Libor Předešlý | `8ae00617-a02d-19e1-2edd-6026b14e2046` |
| Contacts | Karel Pětvaldský | `f134dbd7-91ea-c49d-66c9-5f3e69da202c` |

### Query categories and casing rules

| Category (CZ) | Casing rule | Example |
|---|---|---|
| Přesný název | Exact CRM casing | `"ČEMAT trading, spol. s r.o."` |
| Bez diakritiky | Lowercase | `"cemat trading"` |
| Skloňovaný tvar | Lowercase or sentence-case | `"v čematu"` |
| Překlep / STT chyba | Lowercase | `"šemat trejding"` |
| Neúplný dotaz | Lowercase | `"čemat"` |

Rationale for lowercase on categories 2–5: users rarely type CRM names with correct capitalisation during voice input or casual text entry.

### Full query table

| Category | Record | Query | Expected record_id |
|---|---|---|---|
| Přesný název | ČEMAT | `ČEMAT trading, spol. s r.o.` | `151cd173` |
| Přesný název | ACMARK | `ACMARK s.r.o.` | `82b6c408` |
| Přesný název | Bednář | `Patrik Bednář` | `8d1d638f` |
| Přesný název | Předešlý | `Libor Předešlý` | `8ae00617` |
| Přesný název | Pětvaldský | `Karel Pětvaldský` | `f134dbd7` |
| Bez diakritiky | ČEMAT | `cemat trading` | `151cd173` |
| Bez diakritiky | ACMARK | `acmark s.r.o.` | `82b6c408` |
| Bez diakritiky | Bednář | `patrik bednar` | `8d1d638f` |
| Bez diakritiky | Předešlý | `libor predesly` | `8ae00617` |
| Bez diakritiky | Pětvaldský | `karel petvaldsky` | `f134dbd7` |
| Skloňovaný tvar | ČEMAT | `v čematu` | `151cd173` |
| Skloňovaný tvar | ACMARK | `do acmarku` | `82b6c408` |
| Skloňovaný tvar | Bednář | `Patrika Bednáře` | `8d1d638f` |
| Skloňovaný tvar | Předešlý | `libora předešlého` | `8ae00617` |
| Skloňovaný tvar | Pětvaldský | `karla pětvaldského` | `f134dbd7` |
| Překlep / STT | ČEMAT | `šemat trejding` | `151cd173` |
| Překlep / STT | ACMARK | `axmark s.r.o.` | `82b6c408` |
| Překlep / STT | Bednář | `patrik bednat` | `8d1d638f` |
| Překlep / STT | Předešlý | `libor přeďešlý` | `8ae00617` |
| Překlep / STT | Pětvaldský | `karel petvaldski` | `f134dbd7` |
| Neúplný dotaz | ČEMAT | `čemat` | `151cd173` |
| Neúplný dotaz | ACMARK | `acmark` | `82b6c408` |
| Neúplný dotaz | Bednář | `bednář` | `8d1d638f` |
| Neúplný dotaz | Předešlý | `předešlý` | `8ae00617` |
| Neúplný dotaz | Pětvaldský | `pětvaldský` | `f134dbd7` |

---

## Metrics

**Hit Rate@k** — fraction of queries where the expected `record_id` appears in top-k results.

Evaluated at k = 1, 3, 5.

Thesis Table 5 uses Hit Rate@1 as the primary metric (agent picks the top candidate without user disambiguation).

---

## Output Files

### `eval_results_raw.csv`
One row per (query × mode):

```
category,record_label,query,mode,hit@1,hit@3,hit@5,top1_record_id,top1_name,top1_score
```

### `eval_results_summary.csv`
Aggregated Hit Rate@1 per category and mode — directly maps to thesis Table 5:

```
category,semantic_hit@1,hybrid_hit@1
Přesný název,...,...
Bez diakritiky,...,...
Skloňovaný tvar,...,...
Překlep / STT chyba,...,...
Neúplný dotaz,...,...
Průměr,...,...
```

---

## Script Architecture

```
scripts/eval_rag_search.py
├── Config: load settings via get_settings(), Qdrant URL + tenant
├── _search_semantic(client, collection, embedder, query, limit) → list[dict]
│   └── qdrant_client.search() with query vector only
├── _search_hybrid(rag_service, tenant_id, query, limit) → list[dict]
│   └── TenantRAGService.search() — text filter + vector merged
├── _hit_at_k(results, expected_id, k) → 0|1
├── TEST_CASES: list of (category, record_label, query, expected_record_id)
├── main()
│   ├── Init embedder + Qdrant client + TenantRAGService
│   ├── For each test case × each mode: run search, compute hits
│   ├── Write raw CSV
│   └── Aggregate + write summary CSV
└── __main__ guard
```

---

## Running

```bash
# From project root (Docker stack must be running)
source .venv/bin/activate
python scripts/eval_rag_search.py
```

Outputs `eval_results_raw.csv` and `eval_results_summary.csv` in the current directory.

---

## Assumptions / Constraints

- Qdrant is reachable at the URL configured in `.env` (default `http://localhost:6333`)
- Embedding service (`rag_embedding_base_url`) is reachable; if not, `HashEmbedder` fallback is used — results in that case are not meaningful for a thesis comparison
- `limit=5` used for all searches (k=1, k=3, k=5 evaluated from the same result list)
- No module filter applied — searches across all modules to reflect realistic agent usage
