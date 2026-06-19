# Live Streaming Plan — progressive feedback during request processing

Date: 2026-06-12. Goal: the user should never stare at a blank screen for ~40 s. Within ~1 s of
sending a message they should see *something*, and from then on a live narrative of what the
assistant is doing, ending with the answer streaming in word by word.

Scope covers both repos:
- **talk2api2** (this repo) — FastAPI service that does STT → agent loop → answer
- **rest_coripo** (`site/ai/`) — PHP Restler gateway the CRM browser widget actually talks to

---

## 1. Target UX timeline

| t (typical) | What the user sees | Source event |
|---|---|---|
| 0 s | Their bubble appears immediately (text) / recording stops (audio) | client-side, no server needed |
| ~0.5–2 s | Transcribed text fills their bubble, segment by segment (audio flow) | `transcription.partial` / `transcription.done` |
| ~2 s | "Přemýšlím…" indicator | `agent.started` |
| each agent step | "Hledám kontakt Karel Novák…", "Čtu schůzky…" status line | `agent.tool_call` / `agent.tool_result` |
| final LLM call | Assistant bubble grows token by token | `answer.delta` |
| end | Cards, pending-action confirm button, audio link — same payload as today | `answer.done` + `result` |

## 2. Why it currently takes 40 s with zero feedback

Request path (all hops fully buffer):

```
Browser ──POST──▶ PHP AiCoripo::query/audio (Guzzle, timeout=600, blocking)
                    └──POST──▶ FastAPI /process-input/ | /process-audio/
                                 ├─ faster-whisper STT (seconds, result discarded until the very end)
                                 ├─ agent loop: up to 6 × (LLM completion + CRM/RAG tool call), serial
                                 └─ chat-title LLM call, DB commits
                    ◀── one big JSON after everything finishes
                    └─ PHP then runs processCommand(dry-run) — another CRM round trip
Browser ◀── final JSON; only now is anything rendered
```

The latency is inherent to the serial agent loop (local LLM, multiple iterations). Streaming
doesn't make it faster — it makes the wait *legible*. (Separately, the agent already produces
everything we need progressively; we just throw the intermediate signals away.)

---

## 3. Event protocol (SSE)

Transport: **Server-Sent Events** (`text/event-stream`). One-directional fits the use case, it
passes through proxies that WebSockets fight with, and `fetch()` + `ReadableStream` lets us
consume SSE from a POST (the legacy `EventSource` API is GET-only — we don't need it).

Frame format — every event is one SSE message with a JSON body:

```
event: agent.tool_call
data: {"seq": 4, "tool": "rag_search_tool", "label_cz": "Hledám kontakt 'Karel Novák'…"}

```

### Event types

| event | data payload | notes |
|---|---|---|
| `accepted` | `{chat_id, request_id}` | first frame, sent immediately |
| `transcription.partial` | `{text, segment_index}` | audio flow only; one per whisper segment |
| `transcription.done` | `{text}` | full transcript (same value the final payload carries today) |
| `pipeline.mode` | `{mode: "pending_patch"\|"quick_action"\|"agent"}` | which path handled the input |
| `agent.started` | `{iteration_limit}` | |
| `agent.tool_call` | `{seq, tool, label_cz}` | `label_cz` is a human Czech status line derived from tool + args (e.g. "Hledám v CRM…"); raw args stay server-side |
| `agent.tool_result` | `{seq, tool, status, summary_cz?}` | status from the tool envelope (`ok`, `confirmation_required`, …) |
| `answer.delta` | `{text}` | token/chunk of the final answer, already cleaned (see §4.3) |
| `answer.done` | `{text}` | full final message (authoritative; client replaces accumulated deltas) |
| `result` | *today's full response object* | `action_result`, `cards`, `chat_id`, `chat_name`, `chat_history`, `audio_url`, … |
| `error` | `{status, message_to_user}` | terminal |
| `ping` | `{}` | heartbeat every 15 s so proxies don't idle-close |

Contract rules:
- `result` is **always** the last data frame (after `error` too, when a graceful fallback payload
  exists). Its shape is identical to the current blocking response → the existing FE rendering
  code keeps working unchanged once it reads `result`.
- Events are additive; clients must ignore unknown event types.
- `seq` is monotonically increasing for ordering/debugging.

---

## 4. Phase 1 — talk2api2: emit events (no API change yet)

### 4.1 Event plumbing

New module `app/engine/events.py`:

```python
@dataclass
class StreamEvent:
    type: str
    data: dict[str, Any]

EmitFn = Callable[[StreamEvent], Awaitable[None]]   # None-emit = no-op (legacy path)
```

Thread an optional `emit: EmitFn | None = None` parameter through the existing call chain —
**default `None` keeps every current caller and test working untouched**:

- `pipeline.process_input_core(..., emit=None)` — emits `pipeline.mode`, wraps the agent call,
  emits `result` at the end (built from the dict it already returns).
- `agent.run_agent(..., emit=None)` — emits `agent.started`, `agent.tool_call` (before
  `tool_obj.ainvoke`), `agent.tool_result` (after), and answer deltas (§4.3).
- `/process-audio/` flow — STT segments (§4.2).

Czech status labels: small map in `events.py` from tool name (+ key args) → label, e.g.
`rag_search_tool` → `Hledám '{query}'…`, `crm_query_tool` → `Načítám {module} z CRM…`,
`crm_action_tool` → `Připravuji akci v CRM…`, `get_company_overview` → `Načítám detail firmy…`.

### 4.2 Streaming transcription

`faster-whisper`'s `model.transcribe()` already returns a **generator of segments** — today
`app/core/audio.py:transcribe()` joins it eagerly. Add:

```python
def transcribe_segments(file_path, language) -> Iterator[str]:  # yields segment texts
```

In the route, iterate it inside `asyncio.to_thread`-driven chunks (or run the generator in a
thread and push segments through an `asyncio.Queue`), emitting `transcription.partial` per
segment, then `transcription.done` with the joined text. The CPU-bound decode stays off the
event loop exactly as it is now.

### 4.3 Streaming the final answer (the tricky part)

The agent is a hand-rolled loop: each LLM completion is *either* a `<tool_call>` *or* the final
answer (`<answer>…` or plain text), and Qwen may prepend `<think>…</think>`. So we can't blindly
stream every completion. Strategy — replace `llm.ainvoke` with `llm.astream` plus a small
state machine (`app/engine/answer_stream.py`, pure function, unit-testable):

1. **Hold-back buffer.** Accumulate chunks without emitting until we can classify the output:
   - buffer starts with `<think>` → suppress until `</think>`, then re-classify the remainder;
   - first non-whitespace after think is `<tool_call>` → **suppress entirely** (it's a tool call;
     the full text is parsed exactly as today once the stream completes);
   - first non-whitespace is `<answer>` or anything else → it's the final answer → start
     emitting `answer.delta` (strip the `<answer>` tag, watch for closing `</answer>`).
2. **Tail guard.** Keep the last ~12 chars unemitted so a trailing `</answer>` / `<tool_call>`
   opening never leaks to the client.
3. On stream end, run the **existing** post-processing (`_normalize_final_answer_text`,
   `to_user_message`) on the complete text and send `answer.done` with the authoritative value —
   the client replaces its accumulated deltas with it, so any cleanup differences (European date
   rewriting etc.) self-heal.

Note: deltas are emitted lightly cleaned (no markdown/emoji stripping mid-stream); `answer.done`
is the canonical text. Keep that asymmetry documented in the event table.

### 4.4 New endpoints (additive — blocking endpoints stay)

In `app/api/endpoints.py`:

- `POST /process-input/stream` — same body as `/process-input/`; returns
  `StreamingResponse(event_generator(), media_type="text/event-stream")` with headers
  `Cache-Control: no-store`, `X-Accel-Buffering: no` (disables nginx buffering).
- `POST /process-audio/stream` — same multipart contract as `/process-audio/`.

Implementation pattern: an `asyncio.Queue[StreamEvent]`; the pipeline runs as a task with
`emit = queue.put`; the response generator drains the queue, serializes SSE frames, and emits
`ping` on a 15 s timer. On client disconnect, FastAPI raises in the generator → cancel the
pipeline task (the agent loop checks `task.cancelled()` between iterations); **commit the user
message + whatever was produced before cancelling**, mirroring today's error path.

Voice responses: unchanged — TTS already runs as a background task with a pre-known URL; the
`result` event carries `audio_url` exactly like today (the FE keeps its existing fetch).

### 4.5 Tests

- `answer_stream` state machine: table-driven unit tests (think-block, tool-call suppression,
  `<answer>` with/without closing tag, tag split across chunk boundary, plain text).
- SSE endpoint: stub LLM that streams scripted chunks + stub CRM client; assert the exact event
  sequence and that the terminal `result` equals the blocking endpoint's response for the same
  stubs.
- Regression: blocking endpoints' behavior byte-identical when `emit is None`.

---

## 5. Phase 2 — getting the stream to the browser

The PHP gateway sits in the middle. Two options:

### Option A — PHP streaming passthrough (no infrastructure change)

New entry that bypasses Restler's buffered JSON rendering (Restler can't stream; do what
`audioProxy()` already does — raw `echo` + `exit`):

```
POST /ai/v1/ai/query/stream   → Guzzle ['stream' => true] to FastAPI /process-input/stream
POST /ai/v1/ai/audio/stream   → same for /process-audio/stream
```

PHP loop: `while (!$body->eof()) { echo $body->read(8192); flush(); }` after
`ob_end_clean()`, `ini_set('zlib.output_compression', '0')`, and sending
`Content-Type: text/event-stream`, `X-Accel-Buffering: no`. Web-server config required:
`fastcgi_buffering off;` (nginx) / `proxy_buffering off;` for that location; Apache+mod_php
mostly just needs `flush()`.

- − Pins one PHP-FPM worker per active stream for the whole request (today's blocking call
  already does this, so it's no *worse* — but size the FPM pool: concurrent users ≈ workers).
- − One more buffering layer to mis-configure.
- + No new auth surface; works wherever today's setup works.

### Option B — browser connects directly to FastAPI with a short-lived token (recommended)

1. PHP gets a tiny new route `POST ai/stream-token`: returns
   `{token, stream_base_url}` where `token = base64(payload).hmac_sha256(payload, secret)`,
   payload = `{tenant, user_id, user_name, chat_id?, exp: now+60s, nonce}`. Signed with a
   dedicated `AI_STREAM_TOKEN_SECRET` shared with FastAPI — the per-tenant CRM HMAC secret is
   **never** exposed to the browser.
2. Browser calls FastAPI directly: `POST {stream_base_url}/process-input/stream` with
   `Authorization: Bearer <token>` and consumes the SSE stream.
3. FastAPI: extend `get_tenant_context` to accept the bearer token as a third auth mode
   (validate signature + expiry; token supplies tenant/user, so the `X-Tenant`/`X-User-Id`
   headers come from the token, not the client).

- + No PHP worker pinned during the 40 s; PHP only signs a token (ms).
- + Lowest latency, no intermediate buffering to debug.
- − FastAPI must be reachable from the user's browser (reverse-proxy a public path to it) and
  CORS configured (`cors_allow_origins` already exists).

**Recommendation:** implement Option A first (smallest change, unblocks the UX), design the token
auth so Option B can replace the transport later without touching the event protocol. If the
FastAPI service is already network-reachable from browsers in the target deployment, skip A.

---

## 6. Phase 3 — PHP gateway adjustments around streaming

- **Move the dry-run preview off the critical path.** Today `AiCoripo::query()`/`audio()` run
  `processCommand($cmd, $user, dryRun: true)` *after* the gateway response, adding a second wait
  before the user sees anything. In the streaming flow the FE receives `result` directly; have
  the FE (or a thin PHP endpoint `POST ai/preview-card`) request the dry-run card *after*
  rendering the answer, so the answer is never blocked by the preview. The card pops in when
  ready.
- The existing blocking `ai/query` / `ai/audio` stay as fallback for old clients; mark them
  deprecated once the widget switches.
- `fetchAudio()`'s poll-with-backoff (8 × 150 ms+) can stay, but the FE should only start
  fetching after the `result` event (today it can race the background TTS task).

## 7. Phase 4 — CRM widget (frontend)

- Render the user's message optimistically at send time; for audio, replace a "🎤 …" placeholder
  with `transcription.partial` text as it arrives.
- One status line under the assistant bubble driven by `agent.tool_call` labels; replace with
  streamed `answer.delta` text; on `answer.done` swap in the canonical text; on `result` render
  cards/actions with the existing renderer (payload is unchanged).
- `fetch()` + `ReadableStream` SSE parser (~30 lines, no library needed); reconnect/cleanup on
  navigation; fall back to the blocking endpoint when the stream endpoint 404s (old backend).

## 8. Rollout & fallback

1. Ship Phase 1 behind `stream_endpoints_enabled` setting (default true — endpoints are additive).
2. Ship PHP passthrough + widget behind a CRM-side feature toggle per instance.
3. Fallback is structural: blocking endpoints remain; widget falls back automatically on
   stream-connect failure.
4. Observability: log per-request event counts + time-to-first-event; the existing
   `log_llm_trace` keeps logging the complete interaction at the end (unchanged).

Rough effort: Phase 1 ≈ 2–3 days (answer-stream state machine is the bulk), Phase 2A ≈ 0.5–1 day
+ web-server config, Phase 3 ≈ 0.5 day, Phase 4 ≈ 1–2 days in the widget.

---

## 9. Coripo-side (`rest_coripo/site/ai`) improvement suggestions

Found while reading the gateway for this plan — independent of streaming, ordered by importance:

1. **Security — `verify => false`** on every Guzzle call in `AiService.php` disables TLS
   certificate verification. Make it configurable and default-on outside dev.
2. **Security — CORS in `ai/index.php`** reflects *any* `Origin` while also sending
   `Access-Control-Allow-Credentials: true` — that combination grants any website credentialed
   access to the API for logged-in users. Use an allowlist of known origins.
3. **Security — `display_errors` is enabled** in `ai/index.php` (`ini_set('display_errors', 1)`)
   — leaks stack traces/paths in production. Tie to an env flag.
4. **Security — `command()` TODO**: the route forwards `$command` to `processCommand` with the
   module/action whitelist still marked TODO. Validate schema + whitelist modules/actions before
   dispatching to SugarBeans.
5. **Bug — `$e->getData()` on a generic `\Exception`** (`AiCoripo.php`, audio error path, ~line
   391) — plain exceptions have no `getData()`; the error handler itself fatals. Guard with
   `method_exists` or catch `RestException` separately.
6. **Worker exhaustion — `timeout => 600`** with blocking Guzzle means one PHP-FPM worker is held
   for up to 10 minutes per AI request. Lower it to a realistic ceiling (~120 s) and align the FPM
   pool size; Option B above removes the problem entirely.
7. **Performance — synchronous dry-run after the answer** (see Phase 3): with the current
   blocking flow this directly adds CRM-side latency to every create/update reply.
8. **Logging — full payload dumps on every request**: `print_r($queryResponse, true)` into
   `logInfo` plus `AiLogger::debug` of raw bodies. The `acm/logs/AiServices/` directory shows
   how large these get; they also contain user conversation content (PII). Gate behind a debug
   flag and add log rotation/retention.
9. **Duplication — `query()` vs `audio()`** share ~80 lines of identical dry-run/card-building
   logic; extract a private `buildPreviewCard(array $aiCommand, \User $user): array` used by both
   (and by the future `ai/preview-card` endpoint).
10. **Magic prefix routing** — `str_starts_with($query, 'Vyhledat:')` switches behavior inside
    `query()` including typographic-quote stripping. Make search an explicit parameter or
    endpoint so the contract is visible to the FE.
11. **Audio proxy buffers whole files** — `audioProxy()` loads the WAV into memory and `echo`s
    it. Stream it (`fpassthru`/chunked read) or, with Option B, let the browser fetch
    `/audio/{id}` from FastAPI directly using the same bearer token.
12. **`hydrateChatContext` / chat history duplication** — both `messages` and `chat_history`
    carry the same array in every response; FE should need only one (halves payload size for
    long chats).
