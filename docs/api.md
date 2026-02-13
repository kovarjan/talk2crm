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
  - returns: command response + chat_id + chat history + optional crm_response

- `POST /process-audio/`
  - multipart: `file`, optional `chat_id`/`X-Chat-Id`, optional `context`, locale, `return_voice`
  - transcribes audio then reuses same pipeline behavior

### Search
- `POST /search/`
  - body uses `input_text` as query and optional `scope` (`contacts|accounts|meetings`)
  - returns scoped or aggregate hybrid search results

### RAG ingest
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
