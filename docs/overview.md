# talk2api overview

## What the project does
- Voice/text assistant that turns natural-language CRM requests into structured commands.
- FastAPI service exposes chat, audio, text, and search endpoints; Redis keeps per-chat history.
- LLM agents classify intents, call ReAct tools, and return validated JSON commands for CRM execution.
- Speech pipeline: Whisper STT (preloaded model) → optional correction; TTS planned.
- Search pipeline: sentence-transformer embeddings + FAISS indexes blended with fuzzy matching for CRM lookups.

## Core data flow
- `main.py` FastAPI endpoints:
  - `/process-input` and `/process-audio`: load chat history, run `run_command_pipeline`, persist history, optionally send CRM commands.
  - `/chats/*`: CRUD for chat history in Redis.
  - `/search`: hybrid search over contacts/accounts/meetings indexes.
- `core/pipelines/command_pipeline.py`:
  - Text via STT (if audio) or payload.
  - `ModuleDataExtractor` classifies module/action/parameters.
  - Routes to module agents (meetings, contacts) which use ReAct tools to gather CRM context.
  - Validates final JSON; writes chat logs.
- `core/services/chat_store.py`: Redis helpers for chat lifecycle.
- `core/adapters/crm_api.py`: HMAC-signed POST to CRM gateway; `process_crm_response` formats responses.

## Agents and tools
- `core/agents/module_data_extractor.py`: single-shot extractor (LLM + heuristics) returning module/action/parameters JSON.
- `core/agents/modules/meetings_agent.py`, `contacts_agent.py`: system prompts for CRM actions; executed through `core/services/llm.run_module_agent`.
- `core/services/llm.py`: LangChain Ollama chat client, ReAct agent setup, JSON parsing/normalization (`parse_agent_output`), reasoning toggle.
- `core/services/tools.py` + `core/services/vector_search.py`/`hybrid_search.py`: tool definitions for finding companies/contacts/meetings via vector + lexical search.

## Speech and text services
- `core/services/stt.py`: loads Whisper model from `WHISPER_MODEL` env; transcribes uploads; optional correction agent.
- `core/services/tts.py`: placeholder using `TTS` (model name via `TTS_MODEL`); not wired into API yet.
- `core/services/textnorm.py`, `spellcheck.py`: Czech text normalization/spell-check utilities.

## Embeddings, ingestion, and search
- `core/embedding/embedder.py`: builds embeddings with SentenceTransformer (`all-MiniLM-L6-v2` default), stores FAISS indexes + metadata in `var/vector`.
- `core/ingestion/ingestor.py`: pulls CRM exports via `core.clients.Client` with pagination/watermarks; writes ndjson.gz snapshots under `var/lake`.
- Hybrid search: `core/services/hybrid_search.py` rescoring (vector cosine + RapidFuzz + bias bonuses) with module-specific mappers; used by `/search` and tools.

## Clients and configuration
- `core/clients/client.py` + `core/clients/clients_config.json`: per-tenant API base URLs/keys, shared HMAC signing.
- `core/config/config.py`: env-driven flags (LLM model/temperature/API URL, Whisper/TTS model names, timing/debug toggles, CRM system/instance).
- Env templates: `sample.env` (runtime), `environment.yml`/`environment.base.yml` (conda), `requirements.txt` (pip).

## Project layout
- `main.py`: primary FastAPI app (preferred).
- `main_old.py`: legacy scaffolding.
- `core/` packages:
  - `agents/`: intent extractor + CRM module agents + Pydantic schemas.
  - `pipelines/`: `command_pipeline.py` orchestration.
  - `services/`: LLM, tools, search, STT/TTS, chat store, utility helpers.
  - `adapters/`: CRM API bridge.
  - `clients/`: tenant configs + HMAC client.
  - `ingestion/`, `embedding/`, `utils/`, `domain/`, `config/`.
- `scripts/`: ingestion runner (`scripts.run_ingest`).
- `data/`, `logs/`, `var/`: sample data, logs, vector/index storage.
- `tests/`: pytest suite (LLM parsing, command pipeline, CZ time parsing, STT, integration CRM tool checks).

## Technologies
- FastAPI, Pydantic, LangChain (ReAct agents) with Ollama backend.
- Whisper STT (`openai-whisper`), SentenceTransformers embeddings + FAISS vector store, RapidFuzz for lexical scoring.
- Redis (async) for chat sessions, requests for outbound CRM calls, pytest for tests.
- Python 3, dotenv for configuration management.

## Running and maintenance
- Install deps: `pip install -r requirements.txt` (or conda via `environment.yml`).
- Run API: `uvicorn main:app --reload --port 3000 --host 0.0.0.0` (or `python main.py`).
- Ingest/update vectors: `python -m scripts.run_ingest --client ai-local --config core/clients/clients_config.json`.
- Tests: `python -m pytest` (integration tests require configured CRM/Redis/indexes).
