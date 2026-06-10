"""
LLM Evaluation Suite — talk2crm Bachelor's Thesis Benchmark
======================================================================
Evaluates local Ollama models as the agent against stubbed CRM/RAG backends.

Usage:
    LLM_MODEL=qwen3:8b pytest tests/test_llm_evaluation.py -v -s
    LLM_MODEL=llama3.1:8b LLM_BASE_URL=http://localhost:11434/v1 pytest tests/test_llm_evaluation.py -v -s

Results are appended to out/llm_benchmark.csv after each test.
"""
from __future__ import annotations

import asyncio
import csv
import json
import os
import re
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import get_settings
from app.engine.agent import run_agent
from app.engine.rag import TenantRAGService
from app.engine.tools import build_tools
from app.services.crm_client import CoripoClient

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TENANT_ID = "ai-local"
USER_ID = "1"
USER_NAME = "testuser"
HMAC_KEY_ID = "your-tenant-id"
HMAC_SECRET = "****"
CRM_BASE_URL = "http://localhost:2000/public"
CRM_TOKEN = f"{HMAC_KEY_ID}:{HMAC_SECRET}"

CSV_PATH = PROJECT_ROOT / "out" / "llm_benchmark.csv"
CSV_HEADER = [
    "timestamp",
    "model",
    "scenario_id",
    "scenario_desc",
    "latency_s",
    "iterations",
    "tool_selected",
    "expected_tool",
    "tool_correct",
    "schema_valid",
    "basic_json_valid",
    "basic_json_error",
    "lang_czech",
    "empty_responses",
    "pass_fail",
    "input",
    "output",
    "response_json",
]

_TRUE_VALUES = {"1", "true", "yes", "on", "y", "t"}
_ALLOWED_CRM_ACTIONS = {"create", "update", "patch", "delete"}

_CZECH_RE = re.compile(
    r"[áéíóúůěšČřžýÁÉÍÓÚŮĚŠČŘŽÝčšž]"
    r"|"
    r"\b(jsem|mám|nalezl|připravil|schůzk|úkol|zítra|byl|jsou|nebyly|žádné|našel|"
    r"připraveno|potvrďte|prosím|schůzky|termín|firmy|vytvořen)\b",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Stub data factories
# ---------------------------------------------------------------------------

def _meetings_stub() -> dict:

    dateTomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
    return {
        "records": [
            {
                "id": "aabb1234-0000-0000-0000-000000000001",
                "name": "Ranní schůzka",
                "date_start": f"{dateTomorrow} 09:00:00",
                "status": "Planned",
                "location": "",
            },
            {
                "id": "aabb1234-0000-0000-0000-000000000002",
                "name": "Odpolední schůzka",
                "date_start": f"{dateTomorrow} 14:00:00",
                "status": "Planned",
                "location": "Praha",
            },
        ]
    }


_CONTACT_ID = "dd445566-0000-0000-0000-000000000001"
_CONTACT_RECORD = {
    "id": _CONTACT_ID,
    "first_name": "Jan",
    "last_name": "Novák",
    "name": "Jan Novák",
    "account_name": "Test s.r.o.",
    "email1": "jan.novak@test.cz",
    "phone_mobile": "+420123456789",
}


def _contact_rag_stub() -> list:
    # Single unambiguous contact so the model never asks for disambiguation.
    return [
        {
            "score": 0.98,
            "payload": {
                "module": "contacts",
                "record": _CONTACT_RECORD,
            },
        }
    ]


def _contacts_crm_stub() -> dict:
    # Returned by crm_query_tool when the model cross-checks contacts in CRM.
    return {"records": [_CONTACT_RECORD]}


def _account_rag_stub(account_id: str = "cc112233-0000-0000-0000-000000000001") -> list:
    return [
        {
            "score": 0.95,
            "payload": {
                "module": "accounts",
                "record": {
                    "id": account_id,
                    "name": "Eleman s.r.o.",
                },
            },
        }
    ]


def _empty_crm_stub() -> dict:
    return {"records": []}


async def _smart_crm_side_effect(module: str, action: str, data: dict) -> dict:
    """Returns contextually realistic CRM data based on the requested module.

    Used in B1/D1 so the model gets confirmation when it cross-checks Jan Novák
    via crm_query_tool, preventing it from asking the user for disambiguation.
    """
    m = (module or "").strip().lower()
    if m == "contacts":
        return _contacts_crm_stub()
    if m == "meetings":
        return _meetings_stub()
    return _empty_crm_stub()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _model_name() -> str:
    return os.environ.get("LLM_MODEL") or get_settings().llm_model


def _make_crm_client() -> CoripoClient:
    client = CoripoClient(
        base_url=CRM_BASE_URL,
        token=CRM_TOKEN,
        user_id=USER_ID,
        user_name=USER_NAME,
    )
    client.execute_module_action = AsyncMock(return_value=_empty_crm_stub())
    return client


def _make_rag_service(search_return: list | None = None) -> MagicMock:
    svc = MagicMock(spec=TenantRAGService)
    svc.search.return_value = search_return or []
    return svc


def _is_czech(text: str) -> bool:
    return bool(_CZECH_RE.search(text or ""))


def _debug_json_enabled() -> bool:
    return str(os.environ.get("LLM_EVAL_DEBUG_JSON", "")).strip().lower() in _TRUE_VALUES


def _schema_valid_action(tool_input: dict) -> bool:
    from app.engine.tool_validator import CrmActionToolArgs
    try:
        CrmActionToolArgs(**tool_input)
        return True
    except Exception:
        return False


def _serialize_json(value: object) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except Exception:
        return str(value)


def _validate_basic_tool_json(steps: list[dict]) -> tuple[bool, str]:
    """Basic structural validation for tool calls in intermediate_steps."""
    for idx, step in enumerate(steps):
        if not isinstance(step, dict):
            return False, f"step[{idx}] is not a dict"
        tool = step.get("tool")
        tool_input = step.get("tool_input")
        if not isinstance(tool, str) or not tool.strip():
            return False, f"step[{idx}] missing/invalid tool"
        if not isinstance(tool_input, dict):
            return False, f"step[{idx}] missing/invalid tool_input"

        if tool == "crm_action_tool":
            module = str(tool_input.get("module", "")).strip()
            action = str(tool_input.get("action", "")).strip().lower()
            if not module:
                return False, f"step[{idx}] crm_action_tool missing module"
            if action not in _ALLOWED_CRM_ACTIONS:
                return False, f"step[{idx}] crm_action_tool invalid action={action!r}"
            if "data_json" not in tool_input:
                return False, f"step[{idx}] crm_action_tool missing data_json"
            data_json = tool_input.get("data_json")
            if isinstance(data_json, str):
                try:
                    json.loads(data_json)
                except Exception:
                    return False, f"step[{idx}] crm_action_tool data_json is not valid JSON string"
            elif not isinstance(data_json, (dict, list)):
                return False, f"step[{idx}] crm_action_tool data_json invalid type={type(data_json).__name__}"
        elif tool == "crm_query_tool":
            module = str(tool_input.get("module", "")).strip()
            if not module:
                return False, f"step[{idx}] crm_query_tool missing module"
        elif tool == "rag_search_tool":
            query = str(tool_input.get("query", "")).strip()
            if not query:
                return False, f"step[{idx}] rag_search_tool missing query"
        elif tool == "get_company_overview":
            account_id = str(tool_input.get("account_id", "")).strip()
            if not account_id:
                return False, f"step[{idx}] get_company_overview missing account_id"

    return True, ""


def _log_csv(row: dict) -> None:
    mode = "a"
    write_header = False

    if not CSV_PATH.exists() or CSV_PATH.stat().st_size == 0:
        mode = "w"
        write_header = True
    else:
        try:
            with CSV_PATH.open("r", newline="", encoding="utf-8") as rf:
                existing_header = next(csv.reader(rf), [])
        except Exception:
            existing_header = []
        if existing_header != CSV_HEADER:
            backup_name = f"{CSV_PATH.stem}.legacy.{datetime.now().strftime('%Y%m%d_%H%M%S')}{CSV_PATH.suffix}"
            backup_path = CSV_PATH.with_name(backup_name)
            CSV_PATH.replace(backup_path)
            mode = "w"
            write_header = True

    with CSV_PATH.open(mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADER)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _build_row(
    *,
    model: str,
    scenario_id: str,
    scenario_desc: str,
    input_text: str,
    start_time: float,
    result: dict,
    expected_tool: str,
    tool_correct: bool,
    schema_valid: bool,
    empty_responses: int,
) -> dict:
    steps = result.get("intermediate_steps", [])
    first_tool = steps[0]["tool"] if steps else ""
    output = result.get("output", "")
    basic_json_valid, basic_json_error = _validate_basic_tool_json(steps)
    lang_czech = _is_czech(output)
    pass_fail = "PASS" if (tool_correct and schema_valid and basic_json_valid and lang_czech) else "FAIL"
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model": model,
        "scenario_id": scenario_id,
        "scenario_desc": scenario_desc,
        "latency_s": round(time.monotonic() - start_time, 3),
        "iterations": len(steps),
        "tool_selected": first_tool,
        "expected_tool": expected_tool,
        "tool_correct": tool_correct,
        "schema_valid": schema_valid,
        "basic_json_valid": basic_json_valid,
        "basic_json_error": basic_json_error,
        "lang_czech": lang_czech,
        "empty_responses": empty_responses,
        "pass_fail": pass_fail,
        "input": input_text,
        "output": output,
        "response_json": _serialize_json(result),
    }


def _print_result(scenario_id: str, row: dict, result: dict) -> None:
    print(f"\n{'=' * 60}")
    print(f"[{scenario_id}] model={row['model']}  {row['pass_fail']}")
    print(f"  latency={row['latency_s']}s  iterations={row['iterations']}")
    print(f"  tool_selected={row['tool_selected']!r}  expected={row['expected_tool']!r}")
    print(
        "  "
        f"tool_correct={row['tool_correct']}  "
        f"schema_valid={row['schema_valid']}  "
        f"basic_json_valid={row['basic_json_valid']}  "
        f"lang_czech={row['lang_czech']}"
    )
    if not row["basic_json_valid"] and row.get("basic_json_error"):
        print(f"  basic_json_error: {row['basic_json_error']}")
    print(f"  empty_responses={row['empty_responses']}")
    print(f"  output: {str(result.get('output', ''))[:200]!r}")
    if _debug_json_enabled():
        try:
            payload = json.dumps(result, ensure_ascii=False, indent=2, default=str)
        except Exception:
            payload = str(result)
        print("  response_json:")
        print(payload)
    print(f"{'=' * 60}")

# ---------------------------------------------------------------------------
# Session-level setup
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session", autouse=True)
def ensure_out_dir():
    (PROJECT_ROOT / "out").mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# A1 — Intent Routing: My Meetings
# ---------------------------------------------------------------------------

def test_a1_intent_routing_my_meetings():
    """'Kdy mám zítra schůzky?' → must call my_meetings_tool first."""
    model = _model_name()
    input_text = "Kdy mám zítra schůzky?"

    crm = _make_crm_client()
    crm.execute_module_action = AsyncMock(return_value=_meetings_stub())
    rag = _make_rag_service()

    tools = build_tools(
        tenant_id=TENANT_ID,
        user_id=USER_ID,
        input_text=input_text,
        request_context=None,
        crm_client=crm,
        rag_service=rag,
        action_confirmation=False,
    )

    empty_responses: list[int] = [0]

    import app.engine.agent as _agent_mod
    _orig_warn = _agent_mod.logger.warning

    def _count_warn(msg, *args, **kwargs):
        if "Empty/thinking-only" in str(msg):
            empty_responses[0] += 1
        _orig_warn(msg, *args, **kwargs)

    _agent_mod.logger.warning = _count_warn
    t0 = time.monotonic()
    try:
        result = asyncio.run(
            run_agent(
                tenant_id=TENANT_ID,
                user_id=USER_ID,
                input_text=input_text,
                context=None,
                tools=tools,
            )
        )
    finally:
        _agent_mod.logger.warning = _orig_warn

    steps = result.get("intermediate_steps", [])
    first_tool = steps[0]["tool"] if steps else ""
    tool_correct = first_tool == "my_meetings_tool"

    row = _build_row(
        model=model,
        scenario_id="A1",
        scenario_desc="Intent routing: my meetings",
        input_text=input_text,
        start_time=t0,
        result=result,
        expected_tool="my_meetings_tool",
        tool_correct=tool_correct,
        schema_valid=True,
        empty_responses=empty_responses[0],
    )
    _log_csv(row)
    _print_result("A1", row, result)

    assert tool_correct, (
        f"A1 FAIL: expected my_meetings_tool as first tool, "
        f"got {first_tool!r}. steps={[s['tool'] for s in steps]}"
    )

# ---------------------------------------------------------------------------
# A2 — Intent Routing: Company Search
# ---------------------------------------------------------------------------

def test_a2_intent_routing_company_search():
    """'Najdi informace o firmě Eleman.' → rag_search_tool first, then get_company_overview."""
    model = _model_name()
    input_text = "Najdi informace o firmě Eleman."

    crm = _make_crm_client()
    crm.execute_module_action = AsyncMock(
        return_value={"records": [{"id": "cc112233-0000-0000-0000-000000000001", "name": "Eleman s.r.o."}]}
    )
    rag = _make_rag_service(_account_rag_stub())

    tools = build_tools(
        tenant_id=TENANT_ID,
        user_id=USER_ID,
        input_text=input_text,
        request_context=None,
        crm_client=crm,
        rag_service=rag,
        action_confirmation=False,
    )

    empty_responses: list[int] = [0]
    import app.engine.agent as _agent_mod
    _orig_warn = _agent_mod.logger.warning

    def _count_warn(msg, *args, **kwargs):
        if "Empty/thinking-only" in str(msg):
            empty_responses[0] += 1
        _orig_warn(msg, *args, **kwargs)

    _agent_mod.logger.warning = _count_warn
    t0 = time.monotonic()
    try:
        result = asyncio.run(
            run_agent(
                tenant_id=TENANT_ID,
                user_id=USER_ID,
                input_text=input_text,
                context=None,
                tools=tools,
            )
        )
    finally:
        _agent_mod.logger.warning = _orig_warn

    steps = result.get("intermediate_steps", [])
    tool_names = [s["tool"] for s in steps]
    first_tool = tool_names[0] if tool_names else ""
    rag_first = first_tool == "rag_search_tool"
    overview_called = "get_company_overview" in tool_names
    tool_correct = rag_first and overview_called

    row = _build_row(
        model=model,
        scenario_id="A2",
        scenario_desc="Intent routing: company search",
        input_text=input_text,
        start_time=t0,
        result=result,
        expected_tool="rag_search_tool→get_company_overview",
        tool_correct=tool_correct,
        schema_valid=True,
        empty_responses=empty_responses[0],
    )
    _log_csv(row)
    _print_result("A2", row, result)

    assert tool_correct, (
        f"A2 FAIL: expected rag_search_tool first then get_company_overview, "
        f"rag_first={rag_first}, overview_called={overview_called}, steps={tool_names}"
    )

# ---------------------------------------------------------------------------
# A3 — Intent Routing: Invoice Status Check
# ---------------------------------------------------------------------------

_BANK_ACCOUNT_ID = "ee778899-0000-0000-0000-000000000001"

def _bank_rag_stub() -> list:
    return [
        {
            "score": 0.96,
            "payload": {
                "module": "accounts",
                "record": {
                    "id": _BANK_ACCOUNT_ID,
                    "name": "365.bank",
                },
            },
        }
    ]


def _bank_overview_stub() -> dict:
    return {
        "module": "Accounts",
        "record": {
            "id": _BANK_ACCOUNT_ID,
            "name": "365.bank",
        },
        "related_records": {
            "AOS_Invoices": [
                {
                    "id": "ff001122-0000-0000-0000-000000000001",
                    "name": "INV-2026-001",
                    "billing_account_name": "365.bank",
                    "status": "Sent",
                    "grand_total": "12000.00",
                    "date_due": "2026-03-31",
                },
                {
                    "id": "ff001122-0000-0000-0000-000000000002",
                    "name": "INV-2026-002",
                    "billing_account_name": "365.bank",
                    "status": "Paid",
                    "grand_total": "8500.00",
                    "date_due": "2026-04-15",
                },
            ]
        },
    }


async def _bank_crm_side_effect(module: str, action: str, data: dict) -> dict:
    m = (module or "").strip().lower()
    a = (action or "").strip().lower()
    if m == "accounts" and a == "company_overview":
        return _bank_overview_stub()
    return _empty_crm_stub()


def test_a3_invoice_status_check():
    """'Má firma 365.bank zaplacené všechny faktury?' → rag_search_tool first, then get_company_overview."""
    model = _model_name()
    input_text = "Má firma 365.bank zaplacené všechny faktury?"

    crm = _make_crm_client()
    crm.execute_module_action = AsyncMock(side_effect=_bank_crm_side_effect)
    rag = _make_rag_service(_bank_rag_stub())

    tools = build_tools(
        tenant_id=TENANT_ID,
        user_id=USER_ID,
        input_text=input_text,
        request_context=None,
        crm_client=crm,
        rag_service=rag,
        action_confirmation=False,
    )

    empty_responses: list[int] = [0]
    import app.engine.agent as _agent_mod
    _orig_warn = _agent_mod.logger.warning

    def _count_warn(msg, *args, **kwargs):
        if "Empty/thinking-only" in str(msg):
            empty_responses[0] += 1
        _orig_warn(msg, *args, **kwargs)

    _agent_mod.logger.warning = _count_warn
    t0 = time.monotonic()
    try:
        result = asyncio.run(
            run_agent(
                tenant_id=TENANT_ID,
                user_id=USER_ID,
                input_text=input_text,
                context=None,
                tools=tools,
            )
        )
    finally:
        _agent_mod.logger.warning = _orig_warn

    steps = result.get("intermediate_steps", [])
    tool_names = [s["tool"] for s in steps]
    first_tool = tool_names[0] if tool_names else ""
    rag_first = first_tool == "rag_search_tool"
    overview_called = "get_company_overview" in tool_names
    tool_correct = rag_first and overview_called

    row = _build_row(
        model=model,
        scenario_id="A3",
        scenario_desc="Invoice status check: 365.bank",
        input_text=input_text,
        start_time=t0,
        result=result,
        expected_tool="rag_search_tool→get_company_overview",
        tool_correct=tool_correct,
        schema_valid=True,
        empty_responses=empty_responses[0],
    )
    _log_csv(row)
    _print_result("A3", row, result)

    assert tool_correct, (
        f"A3 FAIL: expected rag_search_tool first then get_company_overview, "
        f"rag_first={rag_first}, overview_called={overview_called}, steps={tool_names}"
    )


# ---------------------------------------------------------------------------
# B1 — Temporal Extraction: Schedule Meeting
# ---------------------------------------------------------------------------

def test_b1_temporal_extraction_schedule_meeting():
    """
    'Naplánuj schůzku s Janem Novákem na příští úterý ve 14:00.'
    Must call crm_action_tool with module=Meetings, action=create,
    and a YYYY-MM-DD HH:MM datetime in data_json.
    """
    model = _model_name()
    input_text = "Naplánuj schůzku s Janem Novákem na příští úterý ve 14:00."

    crm = _make_crm_client()
    # Smart mock: returns Jan Novák when the model cross-checks via crm_query_tool,
    # so it never gets an empty result that would cause it to ask for disambiguation.
    crm.execute_module_action = AsyncMock(side_effect=_smart_crm_side_effect)
    rag = _make_rag_service(_contact_rag_stub())

    tools = build_tools(
        tenant_id=TENANT_ID,
        user_id=USER_ID,
        input_text=input_text,
        request_context=None,
        crm_client=crm,
        rag_service=rag,
        action_confirmation=False,
    )

    empty_responses: list[int] = [0]
    import app.engine.agent as _agent_mod
    _orig_warn = _agent_mod.logger.warning

    def _count_warn(msg, *args, **kwargs):
        if "Empty/thinking-only" in str(msg):
            empty_responses[0] += 1
        _orig_warn(msg, *args, **kwargs)

    _agent_mod.logger.warning = _count_warn
    t0 = time.monotonic()
    try:
        result = asyncio.run(
            run_agent(
                tenant_id=TENANT_ID,
                user_id=USER_ID,
                input_text=input_text,
                context=None,
                tools=tools,
            )
        )
    finally:
        _agent_mod.logger.warning = _orig_warn

    steps = result.get("intermediate_steps", [])
    action_steps = [s for s in steps if s["tool"] == "crm_action_tool"]

    tool_correct = False
    schema_valid = False

    if action_steps:
        ti = action_steps[0]["tool_input"]
        schema_valid = _schema_valid_action(ti)
        module_ok = str(ti.get("module", "")).lower() == "meetings"
        action_ok = str(ti.get("action", "")).lower() == "create"

        dj = ti.get("data_json", "{}")
        if isinstance(dj, str):
            try:
                dj_obj = json.loads(dj)
            except Exception:
                dj_obj = {}
        else:
            dj_obj = dj if isinstance(dj, dict) else {}
        dj_str = json.dumps(dj_obj)
        date_valid = bool(re.search(r"\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}", dj_str))
        tool_correct = module_ok and action_ok and date_valid

    row = _build_row(
        model=model,
        scenario_id="B1",
        scenario_desc="Temporal extraction: schedule meeting",
        input_text=input_text,
        start_time=t0,
        result=result,
        expected_tool="crm_action_tool",
        tool_correct=tool_correct,
        schema_valid=schema_valid,
        empty_responses=empty_responses[0],
    )
    _log_csv(row)
    _print_result("B1", row, result)

    assert tool_correct, (
        f"B1 FAIL: steps={[s['tool'] for s in steps]}, "
        f"action_steps={action_steps}"
    )

# ---------------------------------------------------------------------------
# C1 — Multi-Turn Context Memory: Change Deadline
# ---------------------------------------------------------------------------

def test_c1_multi_turn_context_memory():
    """
    Turn 1: 'Vytvoř úkol zavolat do firmy 365.bank.' → agent confirms.
    Turn 2: 'Změň termín na zítřek.' → agent must call crm_action_tool
            with a date_due matching tomorrow in data_json.
    """
    model = _model_name()
    tomorrow = (datetime.now().date() + timedelta(days=1)).isoformat()

    # ---- Turn 1 ----
    input_t1 = "Vytvoř úkol zavolat do firmy 365.bank."
    crm1 = _make_crm_client()
    rag1 = _make_rag_service()

    tools_t1 = build_tools(
        tenant_id=TENANT_ID,
        user_id=USER_ID,
        input_text=input_t1,
        request_context=None,
        crm_client=crm1,
        rag_service=rag1,
        action_confirmation=False,
    )

    result_t1 = asyncio.run(
        run_agent(
            tenant_id=TENANT_ID,
            user_id=USER_ID,
            input_text=input_t1,
            context=None,
            tools=tools_t1,
        )
    )

    # Extract pending_action from Turn 1 tool observations
    pending_action: dict = {}
    for step in result_t1.get("intermediate_steps", []):
        obs = step.get("observation", "")
        try:
            obs_parsed = json.loads(obs) if isinstance(obs, str) else obs
        except Exception:
            obs_parsed = {}
        if isinstance(obs_parsed, dict) and obs_parsed.get("status") == "confirmation_required":
            pending_action = obs_parsed.get("pending_action") or {}
            break

    # Fallback: build a minimal pending_action if the model took a different path
    if not pending_action:
        due = (datetime.now() + timedelta(days=7)).strftime("%Y-%m-%d")
        pending_action = {
            "module": "Tasks",
            "action": "create",
            "data": {
                "fields": {
                    "name": "Zavolat do firmy 365.bank",
                    "date_due": due,
                    "status": "Not Started",
                }
            },
            "requires_confirmation": True,
        }

    chat_history = [
        {"role": "user", "content": input_t1},
        {"role": "assistant", "content": result_t1.get("output", "")},
    ]
    context_t2 = {"pending_action": pending_action}

    # ---- Turn 2 ----
    input_t2 = "Změň termín na zítřek."
    crm2 = _make_crm_client()
    rag2 = _make_rag_service()

    tools_t2 = build_tools(
        tenant_id=TENANT_ID,
        user_id=USER_ID,
        input_text=input_t2,
        request_context=context_t2,
        crm_client=crm2,
        rag_service=rag2,
        action_confirmation=False,
    )

    empty_responses: list[int] = [0]
    import app.engine.agent as _agent_mod
    _orig_warn = _agent_mod.logger.warning

    def _count_warn(msg, *args, **kwargs):
        if "Empty/thinking-only" in str(msg):
            empty_responses[0] += 1
        _orig_warn(msg, *args, **kwargs)

    _agent_mod.logger.warning = _count_warn
    t0 = time.monotonic()
    try:
        result_t2 = asyncio.run(
            run_agent(
                tenant_id=TENANT_ID,
                user_id=USER_ID,
                input_text=input_t2,
                context=context_t2,
                tools=tools_t2,
                chat_history=chat_history,
            )
        )
    finally:
        _agent_mod.logger.warning = _orig_warn

    steps_t2 = result_t2.get("intermediate_steps", [])
    out = str(result_t2.get("output", "") or "")
    # Only consider create calls — update is wrong per system prompt rules when
    # pending_action.action == "create" (no record id exists yet).
    create_steps = [
        s for s in steps_t2
        if s["tool"] == "crm_action_tool"
        and str(s["tool_input"].get("action", "")).lower() == "create"
    ]

    tool_correct = False
    schema_valid = False

    if create_steps:
        # Use the last create call — the model may have corrected itself.
        ti = create_steps[-1]["tool_input"]
        schema_valid = _schema_valid_action(ti)
        dj = ti.get("data_json", "{}")
        if isinstance(dj, str):
            try:
                dj_obj = json.loads(dj)
            except Exception:
                dj_obj = {}
        else:
            dj_obj = dj if isinstance(dj, dict) else {}
        dj_str = json.dumps(dj_obj, ensure_ascii=False)
        # Require a concrete runtime date for "tomorrow" (YYYY-MM-DD),
        # optionally followed by a time part.
        tomorrow_re = re.compile(
            rf"\b{re.escape(tomorrow)}(?:[ T]\d{{2}}:\d{{2}}(?::\d{{2}})?)?\b"
        )
        mentions_relative_in_dj = bool(
            re.search(r"\bzítra\b|\bzitra\b|\bzítřek\b", dj_str, re.IGNORECASE)
        )
        mentions_relative_in_out = bool(
            re.search(r"\bzítra\b|\bzitra\b|\bzítřek\b", out, re.IGNORECASE)
        )
        date_updated = bool(tomorrow_re.search(dj_str)) or mentions_relative_in_dj or mentions_relative_in_out
        tool_correct = date_updated
    else:
        # Fallback acceptance: model may answer directly without a tool call.
        # In that case, accept an explicit Czech confirmation mentioning
        # relative tomorrow wording.
        mentions_tomorrow = bool(
            re.search(r"\bzítra\b|\bzitra\b|\bzítřek\b", out, re.IGNORECASE)
        )
        is_confirmation = bool(
            re.search(r"\bpotvrďte\b|\bpotvrd(ím|it)\b|\bpřipraveno\b", out, re.IGNORECASE)
        )
        tool_correct = mentions_tomorrow and is_confirmation
        # No tool-call schema to validate in this fallback path.
        schema_valid = tool_correct

    row = _build_row(
        model=model,
        scenario_id="C1",
        scenario_desc="Multi-turn context memory: change deadline",
        input_text=input_t2,
        start_time=t0,
        result=result_t2,
        expected_tool="crm_action_tool",
        tool_correct=tool_correct,
        schema_valid=schema_valid,
        empty_responses=empty_responses[0],
    )
    _log_csv(row)
    _print_result("C1", row, result_t2)
    print(
        "[C1 expected] "
        "either: "
        f"(A) tool='crm_action_tool', action='create', date_due contains tomorrow='{tomorrow}' "
        "or (B) direct Czech confirmation output mentions zítra/zítřek"
    )
    print(
        "[C1 actual] "
        f"output={result_t2.get('output', '')!r}, "
        f"steps={[s.get('tool') for s in steps_t2]}, "
        f"create_steps args={[s.get('tool_input') for s in create_steps]}"
    )

    assert tool_correct, (
        "C1 FAIL: expected either "
        f"(A) tool='crm_action_tool', action='create', date_due contains tomorrow={tomorrow!r} "
        "or (B) direct Czech confirmation output mentions zítra/zítřek; "
        f"actual output={result_t2.get('output', '')!r}, "
        f"steps={[s['tool'] for s in steps_t2]}, "
        f"create_steps args={[s['tool_input'] for s in create_steps]}"
    )

# ---------------------------------------------------------------------------
# D1 — Error Recovery: Correct After Validation Error
# ---------------------------------------------------------------------------

def test_d1_error_recovery():
    """
    Same prompt as B1 but validate_crm_action_call returns an error on the
    first call, then passes. Agent must retry (>=2 crm_action_tool calls)
    or produce a non-empty answer without exhausting the 6-iteration cap.
    """
    model = _model_name()
    input_text = "Naplánuj schůzku s Janem Novákem na příští úterý ve 14:00."

    crm = _make_crm_client()
    crm.execute_module_action = AsyncMock(side_effect=_smart_crm_side_effect)
    rag = _make_rag_service(_contact_rag_stub())

    tools = build_tools(
        tenant_id=TENANT_ID,
        user_id=USER_ID,
        input_text=input_text,
        request_context=None,
        crm_client=crm,
        rag_service=rag,
        action_confirmation=False,
    )

    # Inject a one-shot validation error on the first crm_action_tool call.
    _call_count: list[int] = [0]

    def _patched_validate(*, module: str, action: str, data_json: str) -> str | None:
        _call_count[0] += 1
        if _call_count[0] == 1:
            return "Update/Patch/Delete requires a valid target record id."
        return None

    empty_responses: list[int] = [0]
    import app.engine.agent as _agent_mod
    _orig_warn = _agent_mod.logger.warning

    def _count_warn(msg, *args, **kwargs):
        if "Empty/thinking-only" in str(msg):
            empty_responses[0] += 1
        _orig_warn(msg, *args, **kwargs)

    _agent_mod.logger.warning = _count_warn
    t0 = time.monotonic()
    try:
        with patch("app.engine.tools.validate_crm_action_call", side_effect=_patched_validate):
            result = asyncio.run(
                run_agent(
                    tenant_id=TENANT_ID,
                    user_id=USER_ID,
                    input_text=input_text,
                    context=None,
                    tools=tools,
                )
            )
    finally:
        _agent_mod.logger.warning = _orig_warn

    steps = result.get("intermediate_steps", [])
    action_steps = [s for s in steps if s["tool"] == "crm_action_tool"]
    output = result.get("output", "")

    # Pass: retried the tool at least once (≥2 calls) OR ended with non-empty
    # output before hitting the 6-iteration cap (graceful degradation).
    recovered = len(action_steps) >= 2 or (bool(output) and len(steps) < 6)

    row = _build_row(
        model=model,
        scenario_id="D1",
        scenario_desc="Error recovery: correct after validation error",
        input_text=input_text,
        start_time=t0,
        result=result,
        expected_tool="crm_action_tool (x2)",
        tool_correct=recovered,
        schema_valid=True,
        empty_responses=empty_responses[0],
    )
    _log_csv(row)
    _print_result("D1", row, result)

    assert recovered, (
        f"D1 FAIL: action_tool calls={len(action_steps)}, "
        f"total_iterations={len(steps)}, "
        f"output={output!r}"
    )
