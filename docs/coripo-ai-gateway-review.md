# Coripo AI Gateway Review — `rest_coripo/site/ai`

Date: 2026-06-12. Scope: the PHP side of the AI assistant (`site/ai/` plus the `AiCoripo` routes
mounted in `site/public`), reviewed against the goal of a **robust, clean AI agent system** working
with the CRM. Companion docs: `streaming-plan.md` (§9 has the quick wins; this doc supersedes and
expands it), `improvement-suggestions.md` (Python side).

> **Status (2026-06-12): P0 implemented** in working trees (uncommitted), plus §2.9 nonce replay
> on both sides: §2.1 `command` route protected; §2.2 keystore from env
> (`AI_INBOUND_HMAC_KEYS_JSON` or `AI_GATEWAY_HMAC_KEY_ID`/`_SECRET`, fail-closed); §2.3 auth
> logging removed; §2.4 audio debug copies gated behind `AI_DEBUG_AUDIO=1` + `debug/.htaccess`
> deny; §2.5 CORS allowlist via `AI_ALLOWED_ORIGINS`; §2.7 `display_errors` and exception details
> gated behind developer mode; §2.9 nonce store wired (PHP: APCu/tmp-file; Python:
> `_register_nonce` in `app/core/security.py`).
>
> **⚠️ Action required before/at deploy:**
> 1. **Rotate the `acmark-ai` HMAC secret** — it is in the coripo git history (3+ commits) and in
>    both deployments' configs. Generate a new one, update `AI_GATEWAY_HMAC_KEY_ID`/`_SECRET`
>    (PHP env) and `HMAC_KEYS_JSON`/`CORIPO_TEST_TOKEN` (talk2api2 `.env`).
> 2. Set `AI_ALLOWED_ORIGINS` if any browser calls `/ai/*` cross-origin (the CRM widget is
>    same-origin via `site/public`, so likely none).
> 3. Any caller of `POST /ai/v1/command` must now send valid HMAC headers (it previously worked
>    unauthenticated).
> Still open from P0 surroundings: §2.6 TLS verify, §2.8 ACL checks, §2.10 multipart signing.

File references are relative to `rest_coripo/site/ai/src/`.

---

## 1. Architecture as it exists today

```
                       session auth                          HMAC (ApiAuth)
Browser widget ──────▶ site/public Restler ─┐       ┌─────▶ site/ai Restler
                       AiCoripo:            │       │       AiGateway:  POST command   ◀── (intended for bridge)
                        POST ai/query       │       │       AiGateway:  GET modules*
                        POST ai/audio       │       │       AIExport:   GET export/*   ◀── RAG ingest feeds
                        POST ai/command     │       │
                        GET  ai/audio/{id}  │       │
                        GET  ai/chat/history│       │
                                            ▼       │
                       AiService (Guzzle client) ───┴──▶ talk2api2 FastAPI
                                            │
                       AiGatewayController + moduleActions traits
                                            │
                                            ▼
                       SugarBeans (direct DB writes)
```

### The core architectural problem: **two competing mutation paths**

| | Path A — Python | Path B — PHP |
|---|---|---|
| Trigger | `crm_action_tool` after user confirms in chat | FE confirm button on a preview card → `ai/command` → `processCommand()` |
| Writes via | Coripo public REST (`set/<module>`), HMAC | SugarBeans directly |
| Entity resolution | `EntityResolver` (fuzzy, Czech inflection: "Novákovou"→"Nováková"), `ModuleAdjustmentEngine` defaults, parent links, invitees | `findContactByFullName()` — splits on the first space, **exact match only**; `findAccountByName()` — exact then `LIKE name%` |
| Date handling | `temporal_resolver`, explicit format | `strtotime()` cascade over 6 possible keys (`getDate()`, AiGatewayController), no timezone policy |
| Defaults | duration, date_start, parent_type via adjustments | its own 60-min default, own status whitelist |
| Validation | tool_validator + Coripo REST validation | per-trait ad-hoc checks |

The same confirmed command can produce **different records** depending on which path executes it,
and "schůzka s Novákovou" resolves on Path A but silently loses the participant on Path B. On top
of that, `AiService::mapDataToCommandParts()` (~480 lines, `AiService.php:975–1455`) exists purely
to translate the Python `pending_action` envelope into the PHP command shape — a third copy of the
field semantics that must be kept in sync with both.

**Recommendation R1 (the most important one):** collapse to a single write path — Path A.
- FE confirm sends the original text + `confirm_action: true` context back through `ai/query`
  (this flow already exists in talk2api2: pending action → confirmation → `crm_action_tool`
  executes via Coripo REST, which runs the CRM's own validation).
- `AiGatewayController` + the five `moduleActions` traits shrink to **dry-run preview rendering
  only** (build the card meta from `pending_action.data` — no re-resolution, no bean writes), or
  disappear entirely if the preview card is rendered from the envelope directly.
- `mapDataToCommandParts()` is deleted; the `pending_action` envelope (already normalized by
  `app/domain/contracts.py` on the Python side) becomes the **single wire contract**. Version it
  (`envelope_version: 2`) so both sides can evolve.
- The `external_id` idempotency trick and `tryNotifyRecordCreatedSync` become unnecessary for
  AI-created records (Path A already tells the bridge what it created).

Everything else in this document assumes R1 as the destination but is actionable independently.

---

## 2. Critical security findings (fix before anything else)

1. **Unauthenticated CRM mutations with user impersonation** — `routes/AiGateway.php:29`:
   `command()` is a **`public`** Restler method (public methods bypass `addAuthenticationClass`;
   the code's own comment says `TODO: secure with protected`). It calls
   `Utils::loginUser($user)` where `$user` comes from the **request body**, and `loginUser()`
   (`Utils.php:125`) "authenticates" by checking that the id+username pair exists and is active —
   no credential of any kind. Net effect: anyone who can reach `/ai/v1/command` can act as any
   active CRM user and create/update/delete records.
   *Fix:* make it `protected` (HMAC enforced), and derive the acting user from the signed
   request (key id → tenant → user mapping, or a signed user claim), never from the plain body.

2. **Hardcoded HMAC secret in source** — `restler/ApiAuth.php:34`: the `acmark-ai` secret is
   committed in the `$keyStore` array (the comment says "move to config/DB/vault in production").
   *Fix:* load from env/config, rotate the exposed key everywhere it is configured.

3. **Auth material logged to a file in the webroot** — `restler/ApiAuth.php:~75`:
   `file_put_contents('z_debug.log', print_r([... 'sig' => $sig, 'raw' => $raw ...]))` writes the
   authorization header, signature, nonce and full request body to `z_debug.log` relative to cwd
   (= `site/ai/`, web-served) on **every** authenticated request. The adjacent
   `AiLogger::debug(['auth' => $auth, ...])` does the same into the shared log.
   *Fix:* delete both; never log `Authorization` or raw bodies at info/debug level.

4. **Every voice recording copied to a web-served debug directory** — `AiService.php:312–319`:
   the `copy($audio['tmp_name'], $debugPath)` into `site/ai/debug/` runs **unconditionally for
   every user** (only the subdirectory differs for user 99), with no cleanup. Same pattern again
   in `fetchAudio()` (`AiService.php:446–451`) which writes the fetched WAV to disk **on every
   retry attempt**. That is permanent, web-reachable storage of user voice data (PII).
   *Fix:* gate strictly behind a debug flag + non-webroot path + retention; add `.htaccess`/nginx
   deny for `site/ai/debug/` regardless.

5. **CORS: reflect-any-origin with credentials** — `ai/index.php:11-14`: echoes the request
   `Origin` while sending `Access-Control-Allow-Credentials: true`, which grants any website
   scripted, credentialed access for logged-in users. *Fix:* explicit origin allowlist.

6. **TLS verification disabled on every outbound call** — `verify => false` in all six Guzzle
   call sites in `AiService.php`. *Fix:* config-driven, default on.

7. **Error/stack leakage** — `ai/index.php` enables `display_errors`;
   `ApiRestler::handleThrowable()` returns exception message, file and line to the client (the
   in-code TODO about hiding it in production is commented out). *Fix:* env-gated.

8. **ACL bypass in the PHP write path** — `AiGatewayController::deleteRecord()` only checks
   `assigned_user_id === current_user` (so users can't delete others' records, but admins/teams
   semantics differ from the CRM UI), and the create/update traits never call
   `$bean->ACLAccess(...)` at all. SugarCRM ACL, team security and workflow expectations are
   silently bypassed. *Fix:* call ACL checks before every bean write — or moot via R1.

9. **No nonce replay protection** — `ApiAuth` supports a `$nonceChecker` but it is never wired
   (`null`), so any captured request is replayable for the 5-minute skew window. Mirrors the same
   gap on the Python side (`docs/improvement-suggestions.md` §3). *Fix:* small APCu/Redis TTL set.

10. **Unsigned audio uploads** — `processAudio()` sends identity headers but **no HMAC** (the
    multipart body would need a signing strategy). It currently works because talk2api2 treats
    HMAC as optional; the moment `REQUIRE_HMAC=true` is enabled there, voice input breaks.
    *Fix:* agree on a multipart signing scheme (e.g. sign `timestamp|nonce|user_id|sha256(file)`)
    — coordinate with the Python `require_hmac` rollout.

---

## 3. Correctness bugs

1. **`getModule()` references a property that doesn't exist** —
   `AiGatewayController.php` iterates `$this->allowedModules`, but the class only defines
   `$allowed`. In PHP 8 `foreach (null)` raises a warning and the loop never runs → every parent
   lookup by `related_module` resolves to `''` (except the hardcoded company→Accounts case), so
   "Týká se" linking via module name is silently broken on the PHP path.
2. **`$e->getData()` on a plain `\Exception`** — `AiCoripo.php` audio error path (~line 391):
   generic exceptions have no `getData()`; the error handler itself fatals and the user gets a
   500 instead of the prepared friendly card.
3. **Restler dev-mode flag precedence** — `ApiRestler.php`:
   `(!$sugar_config['developerMode']['restlerDevMode']) ?? false` — the `??` never fires because
   the left side can't be null; with the key missing this also raises undefined-index warnings.
   Should be `!($sugar_config['developerMode']['restlerDevMode'] ?? false)`.
4. **Timezone double-conversion patch-over** — `shouldConvertPreviewDateTimezone()`
   (`AiCoripo.php:791`) is a heuristic compensating for the fact that the two write paths disagree
   about whether datetimes are user-local or DB time. Fixing the contract (R1 + one documented
   timezone rule in the envelope, e.g. "always user-local wall time + tz name") deletes this.
5. **LLM-output scraping in PHP** — `extract_json_object()` (`AiService.php:736`) hunts for JSON
   between the first `{` and last `}` of free text, including `</think>` handling. Since
   talk2api2 now always returns a normalized envelope (`agent_result.pending_action`,
   `message_to_user`), PHP should read **only** structured fields and never parse model text.
   Same for `sanitizeAssistantText()` / `normalizeMessageContent()` re-stripping think-tags and
   markdown that Python already strips (`to_user_message`) — double-cleaning plus an `nl2br()`
   that the FE then has to undo.

---

## 4. Robustness & contract hygiene

- **One response shape.** `normalizeGatewayResponse()` has to guess between `success`/`status`,
  `response`/`data`, string-or-array payloads, and `hydrateChatContext()` checks four locations
  for `chat_id`. Lock the talk2api2 response schema (it is stable now), validate it in one place,
  and fail loudly on mismatch instead of silently defaulting to a "Můžete upřesnit požadavek?"
  question command.
- **Stop shipping the history twice** — every `ai/query`/`ai/audio` response carries identical
  `messages` and `chat_history` arrays; long conversations double their payload.
- **Idempotency** — `Idempotency-Key` is advertised in the CORS allowlist but implemented
  nowhere; the only dedup is meetings-`external_id`. With R1, idempotency belongs to talk2api2
  (chat-scoped pending action confirm is naturally idempotent); without R1, honor the header in
  `processCommand`.
- **`tryNotifyRecordCreatedSync` covers only meetings/calls** — contacts/tasks/notes created via
  the PHP path never reach the assistant's chat context, so a follow-up "uprav ten úkol" can't
  resolve the record. (Disappears with R1.)
- **Restler route cache** — routes live in `site/public/cache/routes.php`; a stale cache after
  deploying new endpoints is a recurring failure mode. Add cache invalidation to the deploy step.
- **Magic `Vyhledat:` prefix** — search-mode is triggered by string-prefix sniffing inside
  `query()` (including typographic-quote stripping). Make it an explicit `mode`/endpoint.
- **Timeouts** — `timeout => 600` (query, audio) pins a PHP-FPM worker per in-flight AI request
  for up to 10 minutes. Bound it to a realistic ceiling and size the FPM pool; the streaming plan
  (Option B) removes the long-held worker entirely.

---

## 5. Code structure & maintainability

1. **`AiCoripo::query()` vs `audio()`** duplicate ~80 lines of dry-run/preview-card building —
   extract `buildPreviewCard(array $aiCommand): array` (also needed standalone for the streaming
   plan's `ai/preview-card`).
2. **The five `moduleActions` traits are near-copies** (`createMeeting`/`createCall`/`createTask`…
   share the structure: validate → defaults → parent resolution → save → related records). If any
   PHP write path survives R1, replace with one generic bean writer driven by a per-module field
   spec (required fields, defaults, enums, parent rules) — the 5×4 `switch` in `processCommand()`
   becomes a two-line table dispatch.
3. **`generateMeta()`** (`AiCoripo.php:622`, ~160 lines) is another per-module field map that
   should live in the same spec.
4. **Naming/leftovers** — `deleteRecord()` calls everything `$meeting` regardless of module;
   commented-out mock wiring (`AiServiceMock`), commented dead code blocks, `TODO` markers on
   security-critical lines. Sweep them.
5. **No tests.** There is no PHP test in `site/ai/`. Minimum viable safety net:
   (a) unit tests for `normalizeCommandShape`/`mapDataToCommandParts` (pure functions, easy) and
   `extract_json_object` while they still exist; (b) an HTTP contract test that replays recorded
   talk2api2 responses through `AiCoripo::query` and asserts the FE payload shape. This is what
   makes every refactor above safe.

---

## 6. Operations & observability

- **Log volume/PII**: `acm/logs/AiServices/` is already **206 MB**; default level is DEBUG
  (`AiLogger::DEFAULT_LEVEL`), and both `AiLogger::debug` and per-route `Logger` instances
  `print_r` entire request/response payloads (full conversations, user identities) on every call.
  Set level via env (INFO in production), redact message content, add rotation/retention, and
  drop the per-user-99 special log files in favor of a debug flag.
- **Metrics**: nothing measures gateway latency vs. bridge latency. Log one structured line per
  request: `{route, duration_total, duration_bridge, status, chat_id, user_id}` — enough to see
  whether slowness is PHP-side (dry-run, logging) or Python-side, and a prerequisite for
  validating the streaming work.
- **Health**: `ai/ping` exists; surface it in monitoring along with talk2api2's `/ping/` so
  "assistant down" alerts distinguish gateway vs. bridge.

---

## 7. Suggested order of execution

| Phase | Items | Effort |
|---|---|---|
| **P0 — security hotfixes** (independent, ship immediately) | §2.1 protect `command` route, §2.2 secret→env + rotate, §2.3 delete auth logging, §2.4 stop audio debug copies + deny webroot access, §2.5 CORS allowlist, §2.7 hide errors | ~1 day |
| **P1 — contract hardening** | §2.6 TLS verify, §2.9 nonce store, §3.1–3.3 bug fixes, §4 one response shape + single history array, log level/PII (§6) | 1–2 days |
| **P2 — single write path (R1)** | FE confirm → `ai/query` with `confirm_action`; demote `processCommand` to preview-only; delete `mapDataToCommandParts`; define envelope version + timezone rule; align with `require_hmac` + multipart signing (§2.10) | 3–5 days, coordinated with talk2api2 |
| **P3 — structure** | preview-card extraction, generic module spec, tests (§5), metrics (§6) | incremental |
| **P4 — streaming** | per `streaming-plan.md` — lands naturally after P2 because the FE then has a single, simple confirm flow | see that doc |

The end state: the PHP layer is a **thin, authenticated proxy + UI adapter** (auth, preview cards,
CRM-native rendering), talk2api2 owns **all** AI reasoning and all CRM mutations through one
validated path, and the contract between them is one versioned envelope instead of three partial
re-implementations of the same field semantics.
