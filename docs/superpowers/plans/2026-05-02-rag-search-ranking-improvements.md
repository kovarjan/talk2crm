# RAG Search Ranking Improvements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve hybrid RAG search Hit Rate by fixing three ranked failure patterns identified in the accuracy benchmark: meeting records crowding out entity records, query tokens not matched against record names specifically, and diacritics-stripped queries missing the text filter entirely.

**Architecture:** All changes are in `app/engine/rag.py` — `TenantRAGService`. Fix 1 adds a `_strip_diacritics()` pure helper. Fix 2 adds a `text_ascii` payload field in ingest and a matching Qdrant text index. Fix 3 adds a `_sort_text_results()` helper that re-ranks text-filter hits by module priority (Accounts→Contacts→Meetings) and name-field specificity. Fix 4 adds a second text-filter pass on `text_ascii` when the query is already ASCII, catching diacritics-stripped user input.

**Tech Stack:** Python 3.11, qdrant-client, unicodedata (stdlib), pytest

---

## File Structure

| Action | Path | Responsibility |
|---|---|---|
| Modify | `app/engine/rag.py` | All four fixes |
| Create | `tests/test_rag_search_ranking.py` | Unit tests for pure helpers |

---

### Task 1: Add `_strip_diacritics()` static method and unit test

**Files:**
- Modify: `app/engine/rag.py` (after `_clean_embed_text` at line ~404)
- Create: `tests/test_rag_search_ranking.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_rag_search_ranking.py`:

```python
from app.engine.rag import TenantRAGService


def test_strip_diacritics_czech_chars():
    assert TenantRAGService._strip_diacritics("Předešlý") == "predesly"
    assert TenantRAGService._strip_diacritics("ČEMAT trading") == "cemat trading"
    assert TenantRAGService._strip_diacritics("Ilona Sekyrová") == "ilona sekyrova"
    assert TenantRAGService._strip_diacritics("Karel Pětvaldský") == "karel petvaldsky"


def test_strip_diacritics_ascii_passthrough():
    assert TenantRAGService._strip_diacritics("tescan group") == "tescan group"
    assert TenantRAGService._strip_diacritics("ACMARK s.r.o.") == "acmark s.r.o."
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_rag_search_ranking.py -v
```
Expected: `AttributeError: type object 'TenantRAGService' has no attribute '_strip_diacritics'`

- [ ] **Step 3: Add `_strip_diacritics()` to `TenantRAGService`**

In `app/engine/rag.py`, add this static method directly after `_clean_embed_text` (currently at line ~400):

```python
@staticmethod
def _strip_diacritics(text: str) -> str:
    """Return ASCII-folded lowercase copy — used for diacritics-tolerant text index."""
    import unicodedata
    return unicodedata.normalize("NFD", text).encode("ascii", "ignore").decode("ascii").lower()
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_rag_search_ranking.py::test_strip_diacritics_czech_chars tests/test_rag_search_ranking.py::test_strip_diacritics_ascii_passthrough -v
```
Expected: 2 passed

---

### Task 2: Store `text_ascii` in ingest payload and create its Qdrant text index

**Files:**
- Modify: `app/engine/rag.py` (`ingest_records` payload dict ~line 367, `_ensure_collection` ~lines 170–206)

- [ ] **Step 1: Write the failing test**

Add to `tests/test_rag_search_ranking.py`:

```python
def test_strip_diacritics_is_used_to_build_text_ascii():
    """Verifies _strip_diacritics produces the value that would be stored as text_ascii."""
    record_text = '{"name": "Libor Předešlý", "account_name": "ČEMAT trading, spol. s r.o."}'
    result = TenantRAGService._strip_diacritics(record_text)
    assert "predesly" in result
    assert "cemat" in result
    assert "ě" not in result
    assert "š" not in result
```

- [ ] **Step 2: Run test to verify it passes (it's a pure function test)**

```bash
pytest tests/test_rag_search_ranking.py::test_strip_diacritics_is_used_to_build_text_ascii -v
```
Expected: PASS (this verifies the helper behaves correctly; the ingest integration is verified by running the benchmark after re-ingestion)

- [ ] **Step 3: Add `text_ascii` to the ingest payload**

In `app/engine/rag.py`, in `ingest_records()`, locate the `models.PointStruct` payload dict (lines ~367–379). Add `"text_ascii"` after `"text"`:

Before:
```python
payload={
    "tenant_id": tenant_id,
    "module": module,
    "record_id": item["record_id"],
    "record_hash": item["record_hash"],
    "modified_at": (
        item["modified_at"].isoformat()
        if item["modified_at"] is not None
        else None
    ),
    "text": item["text"],
    "record": item["record"],
},
```

After:
```python
payload={
    "tenant_id": tenant_id,
    "module": module,
    "record_id": item["record_id"],
    "record_hash": item["record_hash"],
    "modified_at": (
        item["modified_at"].isoformat()
        if item["modified_at"] is not None
        else None
    ),
    "text": item["text"],
    "text_ascii": self._strip_diacritics(item["text"]),
    "record": item["record"],
},
```

- [ ] **Step 4: Add `text_ascii` index in `_ensure_collection()` for new collections**

In `app/engine/rag.py`, in `_ensure_collection()`, locate the `try` block that creates the `text` payload index (lines ~194–205). Add a second index creation after it:

Before:
```python
        try:
            self.client.create_payload_index(
                collection_name=collection_name,
                field_name="text",
                field_schema=models.TextIndexParams(
                    type="text",
                    tokenizer=models.TokenizerType.WORD,
                    lowercase=True,
                ),
            )
        except Exception:
            logger.warning("Could not create full-text index on collection=%s", collection_name)
        self._ensured_collections.add(collection_name)
        logger.info("Created Qdrant collection name=%s", collection_name)
```

After:
```python
        try:
            self.client.create_payload_index(
                collection_name=collection_name,
                field_name="text",
                field_schema=models.TextIndexParams(
                    type="text",
                    tokenizer=models.TokenizerType.WORD,
                    lowercase=True,
                ),
            )
        except Exception:
            logger.warning("Could not create full-text index on collection=%s", collection_name)
        try:
            self.client.create_payload_index(
                collection_name=collection_name,
                field_name="text_ascii",
                field_schema=models.TextIndexParams(
                    type="text",
                    tokenizer=models.TokenizerType.WORD,
                    lowercase=True,
                ),
            )
        except Exception:
            logger.warning("Could not create text_ascii index on collection=%s", collection_name)
        self._ensured_collections.add(collection_name)
        logger.info("Created Qdrant collection name=%s", collection_name)
```

- [ ] **Step 5: Also ensure `text_ascii` index on existing collections**

In `_ensure_collection()`, locate the `if collection_name in existing:` branch (lines ~170–186). Add index creation before `self._ensured_collections.add(collection_name)`:

Before:
```python
        if collection_name in existing:
            try:
                info = self.client.get_collection(collection_name)
                existing_size = info.config.params.vectors.size  # type: ignore[union-attr]
                if existing_size != self.settings.rag_embedding_size:
                    logger.warning(
                        "Qdrant collection vector size mismatch — re-ingest required "
                        "collection=%s existing_size=%s configured_size=%s",
                        collection_name,
                        existing_size,
                        self.settings.rag_embedding_size,
                    )
            except Exception:
                pass
            self._ensured_collections.add(collection_name)
            return
```

After:
```python
        if collection_name in existing:
            try:
                info = self.client.get_collection(collection_name)
                existing_size = info.config.params.vectors.size  # type: ignore[union-attr]
                if existing_size != self.settings.rag_embedding_size:
                    logger.warning(
                        "Qdrant collection vector size mismatch — re-ingest required "
                        "collection=%s existing_size=%s configured_size=%s",
                        collection_name,
                        existing_size,
                        self.settings.rag_embedding_size,
                    )
            except Exception:
                pass
            try:
                self.client.create_payload_index(
                    collection_name=collection_name,
                    field_name="text_ascii",
                    field_schema=models.TextIndexParams(
                        type="text",
                        tokenizer=models.TokenizerType.WORD,
                        lowercase=True,
                    ),
                )
            except Exception:
                pass  # index already exists or collection doesn't support it
            self._ensured_collections.add(collection_name)
            return
```

- [ ] **Step 6: Run full test suite to confirm no regressions**

```bash
pytest tests/ -x -q
```
Expected: all existing tests pass

---

### Task 3: Add `_sort_text_results()` and apply it after the text filter in `search()`

**Files:**
- Modify: `app/engine/rag.py` (new static method after `_strip_diacritics`, and `search()` at ~line 256)
- Modify: `tests/test_rag_search_ranking.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_rag_search_ranking.py`:

```python
def _make_text_result(module: str, name: str) -> dict:
    return {"score": 1.0, "payload": {"module": module, "record": {"name": name}}}


def test_sort_text_results_accounts_before_meetings():
    results = [
        _make_text_result("Meetings", "TESCAN schůzka"),
        _make_text_result("Accounts", "TESCAN GROUP, a.s."),
    ]
    out = TenantRAGService._sort_text_results(results, "TESCAN")
    assert out[0]["payload"]["module"] == "Accounts"
    assert out[1]["payload"]["module"] == "Meetings"


def test_sort_text_results_contacts_before_meetings():
    results = [
        _make_text_result("Meetings", "Předešlý - schůzka"),
        _make_text_result("Contacts", "Libor Předešlý"),
    ]
    out = TenantRAGService._sort_text_results(results, "Libor Předešlý")
    assert out[0]["payload"]["module"] == "Contacts"


def test_sort_text_results_name_match_first_within_module():
    results = [
        _make_text_result("Accounts", "Jiná firma"),       # no name match
        _make_text_result("Accounts", "TESCAN GROUP, a.s."),  # name match
    ]
    out = TenantRAGService._sort_text_results(results, "tescan")
    assert out[0]["payload"]["record"]["name"] == "TESCAN GROUP, a.s."


def test_sort_text_results_stable_for_equal_priority():
    results = [
        _make_text_result("Accounts", "Alpha s.r.o."),
        _make_text_result("Accounts", "Beta s.r.o."),
    ]
    out = TenantRAGService._sort_text_results(results, "gamma")
    # Both have same priority (Accounts, no name match) — order preserved
    assert out[0]["payload"]["record"]["name"] == "Alpha s.r.o."
    assert out[1]["payload"]["record"]["name"] == "Beta s.r.o."
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_rag_search_ranking.py -k "sort_text_results" -v
```
Expected: `AttributeError: type object 'TenantRAGService' has no attribute '_sort_text_results'`

- [ ] **Step 3: Add module-priority constant and `_sort_text_results()` to `TenantRAGService`**

In `app/engine/rag.py`, add the module-level constant immediately before the `class TenantRAGService:` line:

```python
_MODULE_PRIORITY: dict[str, int] = {"Accounts": 0, "Contacts": 1, "Meetings": 2}
```

Then add `_sort_text_results()` as a static method directly after `_strip_diacritics`:

```python
@staticmethod
def _sort_text_results(results: list[dict[str, Any]], query: str) -> list[dict[str, Any]]:
    """Re-rank text-filter hits: entity records before meetings, name matches before body matches."""
    query_tokens = set(query.lower().split())

    def sort_key(r: dict[str, Any]) -> tuple[int, int]:
        payload = r.get("payload") or {}
        module = payload.get("module", "")
        name = ((payload.get("record") or {}).get("name") or "").lower()
        name_match = 0 if any(t in name for t in query_tokens) else 1
        return (_MODULE_PRIORITY.get(module, 99), name_match)

    return sorted(results, key=sort_key)
```

- [ ] **Step 4: Apply `_sort_text_results()` in `search()` after the text filter pass**

In `app/engine/rag.py`, in `search()`, locate the `except Exception: pass  # full-text index may not exist` line (currently ~line 258). Add one line immediately after it:

Before:
```python
        except Exception:
            pass  # full-text index may not exist on old collections

        # --- Vector similarity pass ---
```

After:
```python
        except Exception:
            pass  # full-text index may not exist on old collections
        text_results = self._sort_text_results(text_results, query)

        # --- Vector similarity pass ---
```

- [ ] **Step 5: Run the new tests**

```bash
pytest tests/test_rag_search_ranking.py -k "sort_text_results" -v
```
Expected: 4 passed

- [ ] **Step 6: Run full test suite**

```bash
pytest tests/ -x -q
```
Expected: all pass

---

### Task 4: Add ASCII fallback text filter pass in `search()`

**Files:**
- Modify: `app/engine/rag.py` (`search()` method)
- Modify: `tests/test_rag_search_ranking.py`

- [ ] **Step 1: Write the unit test for the ASCII detection logic**

Add to `tests/test_rag_search_ranking.py`:

```python
def test_strip_diacritics_detects_ascii_query():
    """ASCII queries (no diacritics) should produce an identical stripped form."""
    ascii_query = "libor predesly"
    assert TenantRAGService._strip_diacritics(ascii_query) == ascii_query

    diacritics_query = "Libor Předešlý"
    assert TenantRAGService._strip_diacritics(diacritics_query) != diacritics_query.lower()
```

- [ ] **Step 2: Run test to verify it passes**

```bash
pytest tests/test_rag_search_ranking.py::test_strip_diacritics_detects_ascii_query -v
```
Expected: PASS (pure function test, no Qdrant needed)

- [ ] **Step 3: Add the ASCII fallback filter pass in `search()`**

In `app/engine/rag.py`, in `search()`, locate the line after `text_results = self._sort_text_results(text_results, query)` that was added in Task 3. Insert the ASCII fallback block between that sort line and `# --- Vector similarity pass ---`:

```python
        text_results = self._sort_text_results(text_results, query)

        # --- ASCII fallback text filter (catches diacritics-stripped queries) ---
        # Run only when the query is already pure ASCII — meaning the normal text
        # filter above would have failed to match records that have diacritics in
        # their stored text (e.g. "predesly" vs stored "Předešlý").
        query_ascii = self._strip_diacritics(query)
        if query_ascii == query.lower() and query_ascii:
            try:
                ascii_filter = models.Filter(
                    must=[
                        models.FieldCondition(
                            key="text_ascii",
                            match=models.MatchText(text=query_ascii),
                        )
                    ]
                )
                ascii_points, _ = self.client.scroll(
                    collection_name=collection,
                    scroll_filter=ascii_filter,
                    with_payload=True,
                    with_vectors=False,
                    limit=limit,
                )
                for point in ascii_points:
                    pid = str(getattr(point, "id", "") or "")
                    if pid not in text_ids:
                        text_ids.add(pid)
                        text_results.append({"score": 1.0, "payload": point.payload})
                if ascii_points:
                    text_results = self._sort_text_results(text_results, query)
            except Exception:
                pass  # text_ascii index may not exist on old collections pre-re-ingest

        # --- Vector similarity pass ---
```

- [ ] **Step 4: Run full test suite**

```bash
pytest tests/ -x -q
```
Expected: all pass

- [ ] **Step 5: Re-ingest tenant data to populate `text_ascii` in Qdrant**

The ASCII fallback only works for records that were ingested AFTER Task 2 was deployed. Re-ingest to populate `text_ascii` on existing records:

```bash
curl -s -X POST http://localhost:8011/rag/ingest/ \
  -H "Content-Type: application/json" \
  -H "X-Tenant: ai-local" \
  -H "X-User-Id: benchmark" \
  -d '{"synchronous": true}' | python3 -m json.tool
```

Expected: `{"success": true, "response": {"inserted": N, ...}}`

- [ ] **Step 6: Re-run the benchmark to measure improvement**

```bash
python scripts/eval_rag_search.py
```

Expected improvements vs. baseline (32% semantic / 40% hybrid):
- Přesný název hybrid: 60% → ~80% (module re-ranking fixes TESCAN meeting crowding)
- Bez diakritiky: 40% → ~60% (ASCII fallback catches libor predesly, karel petvaldsky)
- Průměr hybrid: 40% → ~52%

---

## Self-Review

**Spec coverage:**
- ✓ Fix 1 (module priority re-ranking): Task 3 — `_sort_text_results()` with `_MODULE_PRIORITY`
- ✓ Fix 2 (name-field specificity boost): Task 3 — `name_match` in sort key
- ✓ Fix 3 (diacritics normalization in ingest): Task 2 — `text_ascii` payload field
- ✓ Fix 3 (diacritics index): Task 2 — `create_payload_index` for `text_ascii` on new and existing collections
- ✓ Fix 3 (ASCII fallback in search): Task 4 — second filter pass on `text_ascii`
- ✓ No regressions: `pytest tests/ -x -q` after each task

**Placeholder scan:** None found.

**Type consistency:**
- `_sort_text_results(results: list[dict[str, Any]], query: str) -> list[dict[str, Any]]` — called as `self._sort_text_results(text_results, query)` in Tasks 3 and 4 ✓
- `_strip_diacritics(text: str) -> str` — called as `self._strip_diacritics(item["text"])` in Task 2 and `self._strip_diacritics(query)` in Task 4 ✓
- `_MODULE_PRIORITY` referenced inside `_sort_text_results` — defined at module level before the class ✓
- `text_ids` set in `search()` — already exists in the current implementation, `ascii_points` loop adds to it ✓

**Note for implementer:** After Task 2 is deployed but before re-ingest (Task 4 Step 5), existing Qdrant records have no `text_ascii` field, so the ASCII fallback returns no results — it fails silently. The benchmark numbers will only improve after re-ingest.
