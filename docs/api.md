# API Reference (Implementation-Oriented)

Base app: FastAPI in `main.py`.

## Auth headers
Required for most endpoints:
- `X-Tenant: <tenant>`
- `X-User-Id: <user-id>`

Optional HMAC headers (machine-to-machine use):
- `Authorization: HMAC keyId=<kid>, signature=<b64>`
- `X-Timestamp`
- `X-Nonce`

## Endpoint summary

### OpenAPI/Docs
- `GET /openapi.json`
- `GET /swagger`

### Health
- `GET /`
- `GET /ping/`

### Chat lifecycle
- `POST /chats/` create chat id
- `GET /chats/{chat_id}` fetch chat history + metadata
- `GET /chats/user/{user_id_in_path}` list user chats (same user only)
- `PUT /chats/{chat_id}` replace chat history
- `DELETE /chats/{chat_id}` delete chat

### Input processing
- `POST /process-input/`
  - body: `input_text`, optional `chat_id`, optional `context`, optional `return_voice`
  - `context.capabilities`: list of capability ids for this turn (`crm` is implied; `web`, `form`). Omitted = defaults (`crm`, `form`). Unknown ids are ignored and echoed in `unknown_capabilities`. Legacy `context.web_search: true` still enables `web`.
  - `context.form`: `{"editable": true, "prefix": "view", "values": {field: scalar}}` — current values of the form open in the CRM; used by the `form` capability tools.
  - returns: command response + `chat_id` + chat history + `capabilities` (enabled ids) + `unknown_capabilities` + `form_patch` (or `null`)
  - `form_patch`: `{"module", "record", "fields": {name: value | {id, name}}, "lines": {"mode": "append", "line_module": str|null, "rows": [{<editable line column>: value}], "dropped": [{"row", "reason"}]}, "invitees": {}, "sources": [url], "message", "schema": {"sections": [...]}}` — values proposed for the open form; never written by the gateway. `lines.rows` are append-only new rows validated against the schema's `line_items.line_fields` (editable columns only); a product row carries `product_template_id: {id, name}` resolved through `product_lookup_tool`, or `{name}` alone when unresolved.
  - `context.capabilities` may also include `products` (on by default): adds `product_lookup_tool`. The `crm` capability includes `crm_record_detail_tool(module, record_id, include_lines)` for one record with its line rows and totals.

- `POST /process-input/stream` (SSE): same body; events `accepted`, `pipeline.mode`, `agent.started`, `agent.tool_call`, `answer.delta`, `answer.done`, `form_patch` (payload as above, emitted before `result` when present), `result`, `error`.

- `POST /process-audio/`
  - multipart: `file`, optional `chat_id`/`X-Chat-Id`, optional `context`, locale, `return_voice`
  - transcribes audio then reuses same pipeline behavior

### CRM sync callbacks
- `POST /crm/events/record-created/`
  - body: `chat_id`, `module`, `record_id`, optional `record_name`, optional `user_message`, optional `source`
  - use case: CRM confirms user created a pending record outside talk2crm (for example pending Meeting)
  - effect: app appends assistant history event with CRM ID and clears pending-create flow for next turn

### Search
- `POST /search/`
  - body uses `input_text` as query and optional `scope` (`contacts|accounts|meetings`)
  - returns scoped or aggregate hybrid search results

### RAG ingest
- `modules` accepts `ProductTemplates` (aliases `products`, `producttemplates`); the catalog is ingested through the same `list/{module}` path as other modules.
- `POST /rag/ingest/`
  - body supports:
    - `modules` list (module names)
    - `synchronous` (bool)
    - `record_limit` (optional int)
    - `page_size` (optional int)
    - `incremental` (bool, default `true`) -> ingest only records newer than latest already ingested record per tenant+module

### Audio cache
- `GET /audio/{file_id}` returns cached `.wav` from `CACHE_DIR/audio`

## Response behavior notes
- Pipeline response is always wrapped with `success` and `response` object.
- For executable CRM actions, app attempts CRM call unless already fetched data is present.
- If CRM mode is `off`, CRM execution returns `None` and command response is still returned.
- Latest CRM sync event (`record-created`) is used as contextual hint (`record`, `record_module`) when module matches, so follow-up edits can target known CRM ID.
