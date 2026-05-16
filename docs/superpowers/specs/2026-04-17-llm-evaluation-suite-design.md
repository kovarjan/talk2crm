# LLM Evaluation Suite — Design Spec
**Date:** 2026-04-17  
**Purpose:** Bachelor's thesis benchmark for evaluating local Ollama models as the talk2crm LangChain agent.

---

## Goal

Produce a self-contained pytest suite (`tests/test_llm_evaluation.py`) that runs the real `run_agent()` loop against stubbed CRM/RAG backends, evaluates the LLM's decision-making across five scenarios, and appends structured metrics to `out/llm_benchmark.csv` for thesis tables.

---

## Model Switching

Model is selected via environment variables before running pytest:

```bash
LLM_MODEL=qwen3:8b pytest tests/test_llm_evaluation.py
LLM_MODEL=llama3.1:8b LLM_BASE_URL=http://localhost:11434/v1 pytest tests/test_llm_evaluation.py
```

A session-scoped `benchmark_model` fixture reads `LLM_MODEL` and `LLM_BASE_URL` from env (defaulting to current `Settings` values), calls `get_settings.cache_clear()`, and monkeypatches the settings object so `run_agent()` picks up the override without touching `.env`.

---

## Mocking Strategy

All HTTP and vector-search I/O is stubbed at the lowest boundary so `build_tools()` and `run_agent()` run unmodified.

| Boundary | Mock | Return value |
|----------|------|--------------|
| `CoripoClient.execute_module_action` | `AsyncMock` | Scenario-specific minimal JSON (meetings list, confirmation envelope, etc.) |
| `TenantRAGService.search` | `MagicMock` | List with one realistic contact/account payload containing a UUID |

The `CoripoClient` is instantiated with hardcoded test credentials:
- Tenant ID: `ai-local`
- User ID: `28`
- User Name: `jkovar`
- HMAC Key ID: `acmark-ai`
- HMAC Secret: `VYNZrsOfdVB390E+M41dxy5fQ7RjKiaPKtWyrFUrR0MZOvtttrdb0jd0Kk/wGJrC`

---

## Test Scenarios

### A1 — Intent Routing: My Meetings
- **Input (Czech):** `"Kdy mám zítra schůzky?"`
- **Expected tool:** `my_meetings_tool`
- **Stub return:** `{"status":"ok","module":"Meetings","total":2,"summary":"...","cards":[]}`
- **Pass:** `intermediate_steps[0]["tool"] == "my_meetings_tool"`

### A2 — Intent Routing: Company Search
- **Input (Czech):** `"Najdi informace o firmě Eleman."`
- **Expected tool:** `rag_search_tool` or `crm_query_tool`
- **Stub return:** RAG stub returns one account record
- **Pass:** `intermediate_steps[0]["tool"] in {"rag_search_tool", "crm_query_tool"}`

### B1 — Temporal Extraction: Schedule Meeting
- **Input (Czech):** `"Naplánuj schůzku s Janem Novákem na příští úterý ve 14:00."`
- **Expected tool:** `crm_action_tool` (may be preceded by `rag_search_tool`)
- **Stub return:** RAG returns contact with UUID; `crm_action_tool` stub returns `confirmation_required` envelope
- **Pass (all must hold):**
  - any step calls `crm_action_tool`
  - that step's `tool_input["module"] == "Meetings"` (case-insensitive)
  - `tool_input["action"] == "create"`
  - `data_json` contains a datetime matching `YYYY-MM-DD HH:MM` pattern
  - `CrmActionToolArgs(**tool_input)` validates without raising

### C1 — Multi-Turn Context Memory
- **Turn 1 input:** `"Vytvoř úkol zavolat do firmy 365.bank."`  
  Agent runs, stub returns `confirmation_required`; the returned `pending_action` is captured.
- **Turn 2 input:** `"Změň termín na zítřek."` with `pending_action` injected into context and Turn 1 exchange in `chat_history`
- **Expected tool (Turn 2):** `crm_action_tool` with `action=create` and an updated date field
- **Pass:** Turn 2 `intermediate_steps` contains a `crm_action_tool` call whose `data_json` includes a date field different from Turn 1's date

### D1 — Error Recovery
- **Setup:** Same prompt as B1 but `crm_action_tool` stub returns a validation error on the **first** call: `{"status":"tool_validation_error","message":"Update/Patch/Delete requires a valid target record id."}`
- **Second call:** stub returns the normal `confirmation_required` envelope
- **Pass:** agent calls `crm_action_tool` at least twice **or** produces a non-empty Czech final answer without looping to the max-iteration cap (i.e. `len(intermediate_steps) < 6`)

---

## CSV Schema

File: `out/llm_benchmark.csv`  
Mode: append (`"a"`); header written only when file is new/empty.

| Column | Type | Description |
|--------|------|-------------|
| `timestamp` | ISO datetime | Run completion time |
| `model` | str | `LLM_MODEL` env value |
| `scenario_id` | str | `A1`, `A2`, `B1`, `C1`, `D1` |
| `scenario_desc` | str | Short English label |
| `latency_s` | float | Wall-clock seconds for `run_agent()` |
| `iterations` | int | `len(intermediate_steps)` |
| `tool_selected` | str | `intermediate_steps[0]["tool"]` or `""` |
| `expected_tool` | str | Ground truth (pipe-separated if multiple) |
| `tool_correct` | bool | First (or any) tool matches expected |
| `schema_valid` | bool | Pydantic validation passed on action args |
| `lang_czech` | bool | Final answer contains Czech diacritics (á/é/í/ě/š/č/ř/ž/ů/ý) or common Czech words (jsem, mám, nalezl, připravil, schůzk) |
| `empty_responses` | int | Count of empty LLM outputs in the run |
| `pass_fail` | str | `PASS` or `FAIL` — composite of above three bools |

---

## Implementation Notes

- **`lru_cache` bust:** `get_settings.cache_clear()` before monkeypatching; restored in fixture teardown via `autouse` or explicit `yield`.
- **`empty_responses` counter:** monkeypatch `agent.logger.warning` to intercept calls containing `"Empty/thinking-only"`.
- **`out/` directory:** created by a session-scoped autouse fixture before any test runs.
- **Async tests:** `pytestmark = pytest.mark.asyncio` at module level; requires `pytest-asyncio`.
- **No live network:** `CoripoClient.__init__` is not mocked — only `execute_module_action`. The client is constructed with the test credentials but never makes real HTTP calls.
- **Isolation:** each test gets fresh `AsyncMock`/`MagicMock` instances; no shared state between scenarios.

---

## Out of Scope

- Audio transcription (STT) pipeline
- RAG ingestion / Qdrant upsert
- Tenant DB (no SQLAlchemy session needed)
- The `pending_patch` fast-path (tested separately in `test_pending_patch.py`)
