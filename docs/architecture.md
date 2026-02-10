# Architecture Overview

## Goal
`talk2api` is an interface layer between user requests and CRM operations.
The system should keep CRM backend details hidden from LLM-facing logic.

## Layered design

1. API Layer (`main.py`)
- Authenticates tenant/user.
- Handles chat/session lifecycle.
- Calls command pipeline.
- Optionally executes confirmed CRM command.

2. Orchestration Layer (`core/pipelines/command_pipeline.py`)
- Extracts intent/module/action.
- Routes to module agents.
- Validates final command JSON.

3. Agent + Tool Layer (`core/services/llm.py`, `core/services/tools.py`)
- Agents decide what data is needed.
- Tools provide constrained CRM operations and lookup/search.
- Tool inputs are tenant/user scoped.

4. CRM Service Layer (`core/services/crm_service.py`)
- Single facade for CRM read/write operations.
- Keeps tool code independent from backend protocol details.
- Preferred integration point for policy checks (RBAC, field allowlists, audit).

5. CRM Adapter Layer (`core/adapters/*.py`)
- `crm_api.py`: execution mode selector (`off|gateway|direct`).
- `crm_direct.py`: backend-specific implementation:
  - Coripo direct REST.
  - SugarCRM 6.5 REST v4.1.
- `crm_modules.py`: module metadata access through adapter.

## Current extension points

### Add new CRM backend
- Add backend implementation in `core/adapters/crm_direct.py`.
- Route backend in `_backend_for_tenant`.
- Keep public service methods in `CRMService` unchanged.

### Add new module behavior
- Implement module agent in `core/agents/modules/`.
- Register in `_module_agent_router()` in `core/pipelines/command_pipeline.py`.

### Add report/question mode
Recommended structure:
- Add `core/services/report_service.py` for read-only aggregate/report operations.
- Keep tool calls read-only for report mode.
- If code execution is needed, run generated code in a sandbox worker process with:
  - strict timeout,
  - memory limit,
  - network/file restrictions,
  - explicit data contract (input/output schema).

### Add AI server HMAC impersonation
- Keep user-auth exchange in backend adapter/session resolver path.
- Resolve user-scoped CRM session in one place only (adapter/service).
- Do not expose raw credentials to agents/tools.

## Refactor guidelines
- Keep business logic out of tools; tools should parse/validate and call `CRMService`.
- Keep backend-specific behavior out of agents/prompts.
- Keep response normalization in adapters.
- Prefer small facade methods over passing backend details across layers.

## Suggested next hardening
1. Add operation policy layer in `CRMService` (allowed modules/actions per mode).
2. Add audit log for every create/update/delete (tenant, user, module, fields changed).
3. Add strict schema for tool payloads (Pydantic models).
4. Add integration tests per backend (`coripo`, `sugar_v4_1`).
