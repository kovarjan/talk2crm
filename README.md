# talk2api

talk2api is a FastAPI service that exposes chat and audio endpoints for CRM workflows.

## Local setup (pip + venv)

# 🗣️ talk2api

**talk2api** is an open-source voice and natural language assistant that lets users interact with APIs — Currently supports CRM workflows — using simple voice commands.

💬 _“Add a new contact named Jan Novák from Acmark”_  
📇 _“Show deals closing this month.”_  
📞 _“Log a call with Petra about the marketing campaign.”_

---

## 🚀 Features

- 🎙️ Voice command recognition (supports Czech and English)
- 🤖 AI-powered intent detection (via NLP)
- 📇 CRM actions: create/update contacts, deals, meetings, etc.
- 🔗 API integration layer (pluggable)
- 🧪 Test suite with example commands for QA

---

## 🛠️ Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
cp .env.example .env
```

Dependency management uses pip + pip-tools (`requirements.in` -> `requirements.txt`).
Legacy conda files were moved to `docs/legacy/environment.yml` and `docs/legacy/environment.base.yml`.
To update pins, run: `pip-compile requirements.in -o requirements.txt`.

## Run

```bash
make dev
```

```bash
conda activate talk2api
# Ensure you have the correct environment activated
python main.py
uvicorn main:app --reload --port 3000 --host 0.0.0.
# audio recording websites must use https - use nginx proxy same origin

# ingest
python -m scripts.run_ingest --client ai-local --config core/clients/clients_config.json
# test
python -m pytest -s tests/test_find_company_tool_real.py
```

Swagger UI: http://localhost:3000/swagger

Architecture notes: `docs/architecture.md`

## Common commands

```bash
make setup   # install runtime + dev deps
make lint    # ruff
make fmt     # black
make test    # pytest (unit-only by default)
```

To run the full test suite (including heavier integration checks), use:

```bash
pytest tests
```

## Environment variables

Create `.env` from `.env.example`. Key settings:

- `WHISPER_MODEL`: Whisper model name
- `TTS_MODEL`: TTS model name
- `TTS_METHOD`: TTS backend (default `xtts`)
- `TTS_DEVICE`: TTS device (`auto`, `cpu`, `cuda`)
- `LLM_API_URL`: LLM API endpoint
- `LLM_MODEL_NAME`: LLM model identifier
- `LLM_TEMPERATURE`: LLM temperature float
- `SHOW_TIMING`: enable timing logs (`true/false`)
- `DEBUG_LLM`: enable LLM debug logs (`true/false`)
- `DISABLE_REASONING`: disable reasoning (`true/false`)
- `CRM_SYSTEM`: CRM system key
- `CRM_INSTANCE`: CRM instance name
- `CRM_EXECUTION_MODE`: off|gateway|direct (controls CRM writes/reads)
- `CRM_DIRECT_BACKEND`: coripo|sugar_v4_1 (only for direct mode)

Multi-tenant client config:
- `core/clients/clients_config.json`: list of clients/tenants
- Required fields per client: `name`, `api_url`, `api_key_id`, `api_key`, `api_version`
- Direct REST fields per client: `rest_url`, `rest_hmac_key_id`, `rest_hmac_secret`
- Optional REST headers per client: `rest_user_id_header`, `rest_user_name_header`, `rest_tenant_header`
- Optional direct backend selector per client: `crm_direct_backend` (`coripo` or `sugar_v4_1`)
- Optional Sugar v4.1 fields per client:
  - `sugar_rest_url` (e.g. `https://localhost:2000/service/v4_1/rest.php`)
  - `sugar_auth_mode` (`login` | `session` | `resolver`)
  - `sugar_username`, `sugar_password`, `sugar_password_is_md5`, `sugar_application_name`
  - `sugar_session_id` (for `session` mode)
  - `sugar_session_resolver_url`, `sugar_session_resolver_method` (for `resolver` mode; future user-scoped sid/session exchange)
- `REDIS_URL`: Redis connection string
- `CHAT_KEY_PREFIX`, `CHAT_USER_INDEX_PREFIX`, `CHAT_META_PREFIX`: Redis key prefixes
- `CHAT_TTL_SECONDS`: chat history TTL in seconds
- `DEFAULT_TENANT`: default tenant fallback for tools
- `VECTOR_TOP_K`: vector search default top_k
- `CACHE_DIR`: local cache directory

## Legacy

`main_old.py` has been archived to `docs/legacy/main_old.py`.
