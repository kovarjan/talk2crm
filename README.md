# Sugar Voice Bridge v1.0

Multi-tenant FastAPI gateway for voice/text-to-action workflows against SugarCRM 6.5, with LangChain agent orchestration, tenant-scoped RAG (Qdrant), and audio pipeline support.

## Stack

- FastAPI (Python 3.10+)
- LangChain + OpenAI-compatible Qwen endpoint (Ollama/vLLM)
- Qdrant for tenant-scoped retrieval
- faster-whisper for STT
- edge-tts for TTS
- SQLAlchemy + PostgreSQL/SQLite for tenant/chat state

## Project Structure

```text
app/
  main.py
  api/
    endpoints.py
    dependencies.py
    models.py
  core/
    config.py
    security.py
    audio.py
    logging.py
  engine/
    agent.py
    tools.py
    rag.py
  services/
    tenant_manager.py
    crm_client.py
database/
  models.py
  session.py
  migrations/001_init.sql
scripts/
  create_tenant.py
requirements.txt
```

## Quick Start

1. Install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Configure environment:

```bash
cp .env.example .env
```

3. Start API (local, without Docker):

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8010 --reload
```

Qdrant (required for `/rag/*` endpoints):

```bash
docker run -d --name talk2api2-qdrant \
  -p 6333:6333 \
  -v "$PWD/.qdrant_storage:/qdrant/storage" \
  qdrant/qdrant:latest
```

Web UI for Qdrant is available at `http://localhost:6333/dashboard`

4. Seed first tenant:

```bash
python scripts/create_tenant.py \
  --tenant-id demo \
  --name "Demo Tenant" \
  --crm-base-url "http://localhost:2000/service/v4_1/rest.php" \
  --crm-token "REPLACE_TOKEN"
```

`--crm-token` for Sugar v4.1 can be either:
- existing session id string
- JSON credentials string, for example:
`'{"username":"admin","password":"admin","application_name":"sugar_voice_bridge"}'`

Coripo public REST mode (`site/public/index.php`) is also supported:

```bash
python scripts/create_tenant.py \
  --tenant-id ai-local \
  --name "Local Coripo" \
  --crm-base-url "http://localhost:2000/public" \
  --crm-token "YOUR_HMAC_SECRET"
```

For Coripo:
- set `CORIPO_HMAC_KEY_ID` in `.env` (default `acmark-ai`)
- `--crm-token` can be:
`HMAC secret` (plain string), or JSON:
`'{"mode":"coripo_public","hmac_key_id":"acmark-ai","hmac_secret":"...","session_id":"optional-sid","user_id":"28","user_name":"jkovar"}'`
- send `X-User-Id` on API requests (must be Coripo/Sugar user id)
- if Coripo enforces username checks, set `user_name` in tenant token JSON

Coripo FE (`rest_coripo`) bridge wiring for local development:
- in `rest_coripo/.env`: `AI_GATEWAY_URL='http://host.docker.internal:8011'`
- in `rest_coripo/.env`: `AI_GATEWAY_TENANT='ai-local'`
- in `rest_coripo/.env`: `AI_GATEWAY_HMAC_KEY_ID='acmark-ai'` and matching `AI_GATEWAY_HMAC_SECRET`
- in `talk2api2/.env`: `HMAC_KEYS_JSON={"acmark-ai":"<same-secret-as-rest_coripo>"}`
- recreate php container after `.env` change: `docker compose -f docker-compose.yml up -d php`

5. Open API docs:

- `GET /swagger`

## Docker Dev Stack

Runs full local stack with hot-reload API + PostgreSQL + Adminer + Qdrant.

1. Configure env:

```bash
cp .env.example .env
```

2. Start stack:

```bash
docker compose -f docker-compose.dev.yml up --build -d
```

3. Open services:

- API Swagger: `http://localhost:8011/swagger`
- Adminer: `http://localhost:8085`
- Qdrant dashboard: `http://localhost:6333/dashboard`
- Dozzle (log viewer): `http://localhost:8090`

4. Seed tenant (inside API container):

```bash
docker compose -f docker-compose.dev.yml exec api \
  python scripts/create_tenant.py \
  --tenant-id ai-local \
  --name "Local Coripo" \
  --crm-base-url "http://host.docker.internal:2000/public" \
  --crm-token "YOUR_HMAC_SECRET"
```

5. Stop stack:

```bash
docker compose -f docker-compose.dev.yml down
```

Default dev DB credentials:

- host: `localhost`
- port: `5432`
- db: `sugar_voice_bridge`
- user: `postgres`
- password: `postgres`

## Docker Production Stack

Runs API + PostgreSQL + Qdrant with production-oriented defaults (no hot reload, persistent named volumes, container healthchecks).

1. Create production env file:

```bash
cp .env.production.example .env
```

2. Set strong secrets in `.env` (`POSTGRES_PASSWORD`, `TENANT_SECRET_KEY`, `HMAC_KEYS_JSON`, CORS origins).

3. Start production stack:

```bash
docker compose -f docker-compose.prod.yml up --build -d
```

4. Stop production stack:

```bash
docker compose -f docker-compose.prod.yml down
```

## API Compatibility

Implemented endpoints:

- `GET /openapi.json`
- `GET /swagger`
- `GET /`
- `GET /ping/`
- `POST /chats/`
- `GET /chats/{chat_id}`
- `GET /chats/user/{user_id_in_path}`
- `PUT /chats/{chat_id}`
- `DELETE /chats/{chat_id}`
- `POST /process-input/`
- `POST /process-audio/`
- `POST /search/`
- `GET /audio/{file_id}`

Required headers for protected endpoints:

- `X-Tenant`
- `X-User-Id`

Optional machine-to-machine HMAC headers:

- `Authorization: HMAC keyId=<kid>, signature=<b64>`
- `X-Timestamp`
- `X-Nonce`

## Debugging & Transparency

- Structured JSON logs with request id (`X-Request-Id`) and request latency.
- Tenant id and user id are kept explicit throughout route -> service -> tool -> agent flow.
- Chat history persists user/assistant turns with metadata for replay/debug.
- Tool outputs are captured into assistant message metadata.

## Notes

- `app/engine/rag.py` uses deterministic hash embeddings as a lightweight default for local development.
- In production, replace the embedder with a real embedding model and keep the same tenant filter contract.
- Qdrant uses one collection per tenant (`<QDRANT_COLLECTION>__<tenant_id-sanitized>`), which provides hard tenant isolation at storage level.
- If upgrading from older shared-collection mode, re-ingest tenant data (or migrate vectors) so historical vectors are visible in tenant-specific collections.



## Ingest

```bash
curl -X POST 'http://127.0.0.1:8011/rag/ingest/' \
  -H 'Content-Type: application/json' \
  -H 'X-Tenant: ai-local' \
  -H 'X-User-Id: 28' \
  --data '{
    "modules": ["Contacts"],
    "incremental": true,
    "synchronous": true,
    "record_limit": 2000,
    "page_size": 200
  }'
```

### Run Qdrant docker:

```bash
docker run -d --name talk2api2-qdrant -p 6333:6333 \
  -v "/Users/kovarjan/Sites/localhost/playground/opensource/talk2api2/.qdrant_storage:/qdrant/storage" \
  qdrant/qdrant:latest
```
