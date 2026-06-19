# Improvement Suggestions

Review date: 2026-06-12. Scope: whole `app/` + `database/` codebase, focused on performance and structure. The app works; nothing here is required for correctness unless marked. Items are ordered by impact within each section.

> **Status (2026-06-12):** all five steps of §5 "Suggested order of execution" are implemented.
> Remaining open items: §1.7 `get_user_chats` N+1 (still bounded by the SQL limit), §2.5 quick-actions
> dead-path decision, §2.3 `_extract_records`/`_parse_datetime` twins with *different* semantics
> (left alone deliberately), nonce replay store (§3), and `/audio/` tenant scoping (§3).
> Also discovered during implementation: `CRMReadService` is constructed in `build_tools()` but its
> methods are never called in production (only tests exercise them) — decide whether to wire the v2
> read path into `crm_query_tool` or delete the staging.

---

## 1. Performance — highest impact

### 1.1 Blocking sync calls inside the async event loop (critical)

These freeze **every concurrent request** while they run. This is the single biggest performance problem in the codebase.

| Where | What blocks | How long |
|---|---|---|
| `app/api/endpoints.py:1579` | `transcribe(temp_path, ...)` — faster-whisper is CPU-bound, called directly in the async `/process-audio/` route | seconds per audio file |
| `app/engine/rag.py:99-119` | `OllamaEmbedder.embed()` does a **synchronous** `httpx.Client` POST to the embeddings endpoint; called from the search path (`search`, `search_entities`) which runs inside async tools and `/search/` | one HTTP round trip per query |
| `app/engine/rag.py` (all search/count/scroll) | sync `QdrantClient` used from async context (`rag_search_tool`, `/search/`, `/rag/status/`, and the direct `rag_service.client.scroll(...)` in `app/engine/tools.py:634`) | network round trip per call |

**Fix:**
- `input_text = await asyncio.to_thread(transcribe, temp_path, language=user_locale)` — one-line change, immediate win.
- Give the search path an async embed. `aembed()` already exists (`rag.py:121`) and is used for ingest; route searches through it (the tool functions are already `async def`).
- Either switch to `AsyncQdrantClient` (qdrant-client ships it, same API with `await`), or wrap every Qdrant call in `asyncio.to_thread`. The async client is the cleaner end state.

### 1.2 New `httpx.AsyncClient` per CRM request — no connection pooling

`app/services/crm_client.py:291` opens a fresh client (new TCP + TLS handshake) for **every** Coripo call. An agent run easily makes 3–6 CRM calls; each pays the full connection setup.

**Fix:** hold one `httpx.AsyncClient` per `CoripoClient` instance (created lazily, closed via `aclose()` or just process-lifetime), or a module-level client keyed by base URL. `httpx` clients are safe for concurrent use.

### 1.3 Coripo SID is re-negotiated on every API request

`CoripoClient` is constructed per HTTP request (endpoints.py:914, :1610, :1676 …) and the constructor deliberately discards static SIDs in favour of HMAC (`crm_client.py:84-85`). Since `_coripo_session_id` lives on the instance, the first CRM call of **every request** performs an extra `hmac-login` round trip (`_ensure_coripo_sid`).

**Fix:** add a module-level SID cache keyed by `(base_url, hmac_key_id, user_id)` with a TTL (and invalidation on 401), so hmac-login happens once per user/tenant per TTL window instead of once per request. This halves CRM latency for single-call paths.

### 1.4 Sequential awaits that should be parallel

- `/search/` with `scope=all` runs contacts → accounts → meetings searches one after another (`endpoints.py:1635-1637`), then RAG. Use `asyncio.gather`.
- `CoripoClient.generic_search` scope=all awaits each module inside a dict comprehension (`crm_client.py:787-794`) — 7 sequential round trips. Same fix.

### 1.5 Chat title generation blocks the first response

`_ensure_chat_name` (endpoints.py:1080) runs an LLM call (with a *second* fallback model retry on failure) inline before `/process-input/` returns — exactly on the first message of a chat, where perceived latency matters most.

**Fix:** return the fallback name immediately and generate the proper title in a `BackgroundTask` (commit it separately). Also `max_tokens=200` is generous for a 6-word title; 40 is plenty.

### 1.6 Redundant DB round trips per `/process-input/`

Per request the route currently issues:
1. tenant select in `get_tenant_context` → `assert_user_access` → `get_tenant`
2. the **same** tenant select again in `TenantManager.get_credentials` (endpoints.py:913)
3. `_load_latest_pending_action` — last assistant message
4. `_load_latest_crm_record_created_event` — last **30** assistant messages with full JSON metadata
5. `_load_messages` (full history) for the agent — then sliced to last 10 in Python
6. `_load_messages` again (full history) after appending, for the response payload

**Fixes:**
- Cache tenant rows in `TenantManager` with a short TTL (30–60 s). CLAUDE.md already claims it "caches per-tenant config" — the code doesn't; make the doc true. This also removes the duplicated select and a Fernet decrypt per request.
- Fetch the last ~30 assistant messages **once** and derive both the pending action and the created-record event from that one result set (3 + 4 collapse into one query).
- `_load_messages` for the agent should `LIMIT 10` at SQL level (order desc, reverse in Python). The second full load can be avoided by appending the two new messages to the already-loaded list instead of re-querying.

### 1.7 `get_user_chats` N+1

Known (the NOTE at endpoints.py:1270 says so). One `selectinload(Chat.messages)` or a single `ChatMessage` query with `chat_id IN (...)` removes up to 50 sequential queries. Worth doing now — it's a small change and this endpoint backs the chat list UI.

### 1.8 `ChatOpenAI` instantiated per call

`run_agent` (agent.py:249) and title generation (endpoints.py:851) build a new `ChatOpenAI` — and therefore a new underlying OpenAI client + connection pool — per request. Cache one instance per `(base_url, model)` (an `lru_cache`d factory is enough; the object is stateless across calls).

### 1.9 Indexes for the hot queries

`chat_messages` has single-column indexes on `chat_id`, `tenant_id`, `user_id`, but every hot query filters `(chat_id, tenant_id, user_id)` and orders by `created_at`. Add a composite index `(chat_id, created_at)` (the rest are redundant given chat scoping) and `(tenant_id, user_id, updated_at)` on `chats` for `get_user_chats`.

---

## 2. Structure & architecture

### 2.1 Split `app/api/endpoints.py` (1 774 lines)

It currently mixes five responsibilities. Suggested split — no behaviour change, just moves:

| New module | Take from endpoints.py |
|---|---|
| `app/services/chat_service.py` | `_create_chat`, `_append_message`, `_load_messages`, `_get_chat_or_404`, `_load_latest_pending_action`, `_load_latest_crm_record_created_event` |
| `app/engine/pipeline.py` | `_process_input_core` + the context-merging helpers (`_merge_context_with_pending_action`, `_merge_context_with_created_record`, `_with_soft_ui_focus_hint`, `_is_read_only_data_query`) |
| `app/presentation/agent_result.py` | `_normalize_agent_result_for_ui`, `_extract_pending_action_from_agent_result`, `_to_user_message`, `_looks_like_noise` |
| `app/services/chat_titles.py` | `_generate_chat_name_with_llm`, `_ensure_chat_name`, `_sanitize_chat_name`, `_fallback_chat_name`, `_history_for_title_prompt` |

endpoints.py then shrinks to route definitions + request/response shaping (~400 lines), and the pipeline becomes unit-testable without FastAPI.

### 2.2 Split `app/engine/tools.py` (1 900 lines)

~1 400 lines are free helper functions; the 5 tools are closures at the bottom. Natural seams:

- `app/engine/record_extract.py` — `_extract_records`, `_flatten_record`, `_merge_records_by_id`, `_dedupe_and_limit`, record-name helpers
- `app/engine/account_resolution.py` — `_resolve_accounts_from_qdrant`, `_select_account_candidates`, contact/account enrichment helpers
- `app/presentation/cards.py` (already exists) — `_meeting_cards`, `_contact_cards`, `_generic_cards`, `_record_card`, `_table_card`, `_crm_detail_link`
- `app/engine/date_filters.py` — `_next_week_range`, `_build_meetings_date_filter`, `_build_login_user_meetings_window_filter`, `_sort_records_by_datetime`

### 2.3 Deduplicate copy-pasted helpers (correctness risk: they will drift)

| Helper | Duplicated in | Move to |
|---|---|---|
| `_format_european_dates`, `_THINK_TAG_RE`, `_ANSWER_TAG_RE` cleanup | `endpoints.py:82` and `agent.py:40` | `app/utils/text.py` (exists, nearly empty) |
| Module canonical map (Contacts/Accounts/… incl. the `"opportunites"` typo-alias) | `endpoints.py:585`, `crm_client.py:313`, `crm_client.py:768`, RAG aliases in `rag.py` | new `app/utils/modules.py` |
| `_extract_records` recursive walker | `crm_client.py:446` and `tools.py:114` | one shared implementation |
| `_expand_fuzzy_token_variants`, `_fuzzy_tokens`, `_score_lexical_fuzzy` | `rag.py:703-786` and `tools.py:419-483` | `app/utils/fuzzy.py` |
| `_parse_datetime` | `rag.py:964` and `tools.py:64` | `app/utils/text.py` or `dates.py` |

The fuzzy-scoring duplication is the most dangerous: search-ranking behaviour was recently tuned (see tests), and tuning one copy silently leaves the other stale.

### 2.4 Stop reaching into RAG service internals

`tools.py:627-634` uses `rag_service._tenant_collection(...)` and `rag_service.client.scroll(...)` directly. Add a public method (e.g. `TenantRAGService.scroll_module(tenant_id, module, ...)`) so the Qdrant client and collection naming stay encapsulated (and can be made async in one place, see 1.1).

### 2.5 Remove or revive dead paths

- **Quick actions**: `quick_action_enabled=False` *and* `quick_action_min_confidence=1.0` is documented as intentionally unreachable (config.py:87-92). Two kill switches for one dead feature is confusing — either delete `app/engine/quick_actions.py` + `app/nlu/command_parser.py` + the pipeline branch, or keep just `quick_action_enabled` as the single switch.
- `Settings.llm_temperature` is never used — `run_agent` hardcodes `temperature=0` (agent.py:254). Use the setting or drop it.
- `_create_chat`'s 5-attempt `OperationalError` retry with backoff (endpoints.py:187-197) is a SQLite-era leftover; the project is PostgreSQL-only now (enforced by the `database_url` validator). A plain commit is enough.
- `synthesize_to_file_sync` (audio.py:185) appears unused.

### 2.6 Migrations: replace `create_all` + inline DDL with Alembic

`database/session.py:28-53` hand-rolls a one-off column migration inside `init_db`. This pattern doesn't scale past the second migration and runs DDL on every boot. Alembic gives ordered, reversible migrations and removes the startup DDL cost.

### 2.7 Documentation drift (CLAUDE.md / README)

- CLAUDE.md says the agent uses `create_tool_calling_agent` + `AgentExecutor` and an `OllamaPayloadCallback`. The actual `agent.py` is a hand-rolled prompt loop parsing `<tool_call>` tags (deliberately, per the LiteLLM comment at agent.py:246). Update the doc — this is the first thing a new contributor reads.
- CLAUDE.md describes `SugarClient` with `is_v4_1=True` Sugar mode; the class is `CoripoClient` and the v4.1 path no longer exists (`is_v4_1` is hardcoded `False`, crm_client.py:44).
- CLAUDE.md says `TenantManager` "caches" config — it doesn't (yet; see 1.6).

---

## 3. Security & robustness

- **HMAC is optional** (`dependencies.py:32-33`): a request *without* an `Authorization: HMAC ...` header skips signature verification entirely — `X-Tenant` + `X-User-Id` headers alone are accepted. The trust assumption is documented in `tenant_manager.py:94-100`, but it's one missing reverse-proxy rule away from an open API. Add a `require_hmac: bool` setting that prod environments turn on, so the gateway assumption is enforced in code.
- **Nonce replay**: `verify_hmac_request` checks timestamp skew but never stores nonces, so a captured request can be replayed for `hmac_max_skew_seconds` (300 s). A small TTL set (in-process dict or Redis) closes it.
- **CORS**: `allow_origins=["*"]` together with `allow_credentials=True` (main.py:49-55) is an invalid combination — browsers reject `*` when credentials are on. Configure explicit origins for prod, or drop `allow_credentials`.
- **`/audio/{file_id}`** (endpoints.py:1767) has no tenant context — any party who learns/guesses a 32-hex id can fetch the synthesized audio. Low risk (unguessable ids), but it's the only unauthenticated data endpoint; consider requiring the tenant headers and prefixing filenames with the tenant id. Also: nothing ever deletes old WAVs from `cache/audio` — add cleanup (e.g. delete files older than N hours on startup or a periodic task).

---

## 4. Repo hygiene (quick wins)

- Untracked local clutter in the repo root: `hello.mp3`, `hello2.mp3`, `ollama_chat.log`, `sugar_voice_bridge.db{,-shm,-wal}` (SQLite remnants), `out/`, `logs/`, `cache/`. None are tracked by git, but `.gitignore` doesn't cover several of them — add `*.log`, `*.db*`, `cache/`, `out/`, `logs/`, `*.mp3` so they can't be committed accidentally.
- `tests/` has 23 files / 84 tests but no `conftest.py` visible at root level and no CI config in the repo — a minimal GitHub Actions workflow running `pytest` would protect the refactors above.
- Pin a `ruff` (or at least `ruff check --select I,F`) config — the codebase style is consistent enough that automated enforcement is cheap now.

---

## 5. Suggested order of execution

1. **One-liners first** (an afternoon): `asyncio.to_thread` around `transcribe`; `asyncio.gather` in `/search/` and `generic_search`; `LIMIT` in `_load_messages` for the agent; gitignore additions.
2. **Connection reuse** (a day): shared `httpx.AsyncClient` in `CoripoClient`; SID cache; cached `ChatOpenAI` factory; `TenantManager` TTL cache.
3. **Async RAG** (a day): route search through `aembed`, adopt `AsyncQdrantClient`, add the public scroll method.
4. **Refactor splits** (incremental, behind the existing test suite): endpoints.py split → tools.py split → helper deduplication.
5. **Infrastructure**: Alembic, CI workflow, `require_hmac` flag, CORS/prod config, audio cache cleanup.

Steps 1–3 need no structural change and should noticeably cut p50 latency (one hmac-login + several TCP handshakes removed per request) and fix p99 collapse under concurrent audio/RAG load. Step 4 is what makes the codebase "cleaner" — do it after 1–3 so the perf changes don't conflict with moved code.
