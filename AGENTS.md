# Repository Guidelines

## Project Structure & Module Organization
- `main.py` exposes the FastAPI app plus chat/audio endpoints; `main_old.py` holds legacy scaffolding.
- Core logic lives in `core/`: `agents/` for LLM/assistant behaviors, `pipelines/` for command execution, `services/` for chat state + search, `adapters/` for CRM/API bridges, `ingestion/` and `embedding/` for vector setup, and `utils/` for shared helpers.
- Tests are in `tests/` (pytest), with data samples under `data/` and ingestion scripts in `scripts/`.
- Environment templates: `sample.env` and `environment.yml` (conda); `requirements.txt` lists runtime/test deps.

## Build, Test, and Development Commands
- Install deps: `pip install -r requirements.txt` (or `conda env create -f environment.yml` then `conda activate talk2api`).
- Run API locally: `uvicorn main:app --reload --port 3000 --host 0.0.0.0` (or `python main.py` for the default runner).
- Ingest/update vectors: `python -m scripts.run_ingest --client ai-local --config core/clients/clients_config.json` (adjust client/config for your tenant).
- Quick connectivity check: `bash test_connect_coripo.sh` (uses env vars for CRM connectivity).

## Coding Style & Naming Conventions
- Python, PEP 8, 4-space indent; prefer type hints and small, testable functions.
- Keep FastAPI schemas in place and prefer `pydantic` models for request/response validation.
- Use descriptive module/function names mirroring pipeline stages (e.g., `*_pipeline.py`, `*_service.py`, `*_adapter.py`).
- Stick to snake_case for files/functions, PascalCase for classes, and UPPER_SNAKE for constants/env keys.

## Testing Guidelines
- Framework: pytest. Run all tests with `python -m pytest`. Use `-s` for verbose logs when debugging voice/LLM flows.
- Test files follow `tests/test_*.py`; mirror module names when adding coverage.
- Real CRM/search tests (`test_*_real.py`) expect configured API keys, indexes, and Redis; skip or mark as integration when unavailable.

## Commit & Pull Request Guidelines
- Commit messages in history often start with a bracketed tag (e.g., `[NEW]`) plus a concise description; continue this style when possible.
- Keep commits focused and runnable (tests pass, lint clean if used). Note any integration tests you skipped and why.
- PRs should include: purpose/issue link, summary of user-facing changes, test commands run, and screenshots/logs for API or ingestion changes where relevant.

## Configuration & Security Tips
- Copy `sample.env` to your local `.env`; keep secrets out of git. Ensure Redis/CRM endpoints match your tenant config in `core/clients/clients_config.json`.
- Large artifacts (audio, embeddings) belong in `data/` or `var/` but should not be committed; document external storage locations if required.
