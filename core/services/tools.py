# core/services/tools.py

import json
import os
from typing import Any, Dict
import logging
import datetime

from langchain.agents import Tool

from core.services.company_lookup import find_company_by_name_or_city
from core.services.contacts_lookup import find_contact_by_query
from core.services.company_contacts import list_contacts_for_account, list_contacts_for_company_name
from core.services.meetings_lookup import search_meetings, get_user_agenda, check_user_conflict

# Remove lax defaults for security; keep only for explicit fallbacks if you really want them.
DEFAULT_TENANT = os.getenv("DEFAULT_TENANT", "ai-local")
DEFAULT_TOP_K = int(os.getenv("VECTOR_TOP_K", "5"))

# --- Logging (unchanged) ---
log_dir = "./logs"
os.makedirs(log_dir, exist_ok=True)
log_filename = os.path.join(log_dir, f"{datetime.datetime.now().strftime('%Y-%m-%d')}_tools_calls.log")
logging.basicConfig(filename=log_filename, level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


# ------------------------- Strict payload parsing --------------------------

def _error(msg: str) -> str:
    out = {"success": False, "error": msg}
    logging.error(msg)
    return json.dumps(out, ensure_ascii=False)

def _get_tenant(d: Dict[str, Any]) -> str:
    t = str(d.get("tenant", "")).strip()
    if not t or t.lower() in ("unknown", "none", "null"):
        raise ValueError("Missing or invalid 'tenant'.")
    return t

def _parse_payload(input_str: str) -> dict:
    """
    Accepts either a plain string or a JSON string. Returns dict with:
      - query (str | "")
      - tenant (str)  # required; injected by llm.py wrapper
      - top_k (int)
      - passthrough keys (account_id, day, user_id, date_from, date_to, ...)
    SECURITY: If tenant is missing, we raise -> caller sees {"success": false, ...}
    """
    if not isinstance(input_str, str):
        raise ValueError("Input must be a string.")

    s = input_str.strip()
    data: Dict[str, Any] = {}
    if s.startswith("{") and s.endswith("}"):
        data = json.loads(s)
    else:
        # plain string query; tenant must still be injected by wrapper;
        # keep query in place so tools can still work if wrapper adds tenant.
        data = {"query": s}

    # Validate tenant presence (wrapper in llm.py injects it)
    data["tenant"] = _get_tenant(data)
    # Normalize top_k
    data["top_k"] = int(data.get("top_k", DEFAULT_TOP_K))
    return data


# ------------------------------ Tool functions --------------------------------

def list_contacts_by_company_tool(input_str: str) -> str:
    try:
        logging.info(f"list_contacts_by_company_tool called with input: {input_str}")
        args = _parse_payload(input_str)
        tenant = args["tenant"]

        if args.get("account_id"):
            contacts = list_contacts_for_account(tenant, args["account_id"])
            out = {"success": True, "company": {"id": args["account_id"]}, "contacts": contacts, "tenant": tenant}
        else:
            res = list_contacts_for_company_name(args.get("query", ""), tenant=tenant)
            out = {"success": True, "tenant": tenant, **res}
    except Exception as e:
        out = {"success": False, "error": str(e)}
        logging.error(f"list_contacts_by_company_tool failed: {e}")

    logging.info(f"list_contacts_by_company_tool output: {out}")
    return json.dumps(out, ensure_ascii=False)


def find_company_tool(input_str: str) -> str:
    try:
        logging.info(f"find_company_tool called with input: {input_str}")
        args = _parse_payload(input_str)
        hits = find_company_by_name_or_city(args.get("query", ""), tenant=args["tenant"], top_k=args["top_k"])
        out = {"success": True, "tenant": args["tenant"], "query": args.get("query", ""), "items": hits}
    except Exception as e:
        out = {"success": False, "error": str(e)}
        logging.error(f"find_company_tool failed: {e}")

    logging.info(f"find_company_tool output: {out}")
    return json.dumps(out, ensure_ascii=False)


def find_contact_tool(input_str: str) -> str:
    try:
        logging.info(f"find_contact_tool called with input: {input_str}")
        args = _parse_payload(input_str)
        hits = find_contact_by_query(args.get("query", ""), tenant=args["tenant"], top_k=args["top_k"])
        out = {"success": True, "tenant": args["tenant"], "query": args.get("query", ""), "items": hits}
    except Exception as e:
        out = {"success": False, "error": str(e)}
        logging.error(f"find_contact_tool failed: {e}")

    logging.info(f"find_contact_tool output: {out}")
    return json.dumps(out, ensure_ascii=False)


def find_meetings_tool(input_str: str) -> str:
    """
    Input: JSON {"query": string, "top_k"?: int, "date_from"?: "YYYY-MM-DD HH:MM", "date_to"?: "YYYY-MM-DD HH:MM", "tenant": string}
    Output: {"success": true, "tenant": "...", "meetings":[...]}
    """
    try:
        logging.info(f"find_meetings_tool called with input: {input_str}")
        args = _parse_payload(input_str)
        meetings = search_meetings(
            query=args.get("query", ""),
            top_k=args["top_k"],
            date_from=args.get("date_from"),
            date_to=args.get("date_to"),
            tenant=args["tenant"],             # <-- pass tenant to service
        )
        out = {"success": True, "tenant": args["tenant"], "meetings": meetings}
    except Exception as e:
        out = {"success": False, "error": str(e)}
        logging.error(f"find_meetings_tool failed: {e}")

    logging.info(f"find_meetings_tool output: {out}")
    return json.dumps(out, ensure_ascii=False)


def get_user_agenda_tool(input_str: str) -> str:
    """
    Input: JSON {"day": "YYYY-MM-DD", "user_id"?: string, "tenant": string}
    Output: {"success": true, "tenant":"...", "agenda":[...]}
    """
    try:
        logging.info(f"get_user_agenda_tool called with input: {input_str}")
        args = _parse_payload(input_str)
        agenda = get_user_agenda(
            day=args.get("day", ""),
            user_id=args.get("user_id"),
            tenant=args["tenant"],             # <-- pass tenant to service
        )
        out = {"success": True, "tenant": args["tenant"], "agenda": agenda}
    except Exception as e:
        out = {"success": False, "error": str(e)}
        logging.error(f"get_user_agenda_tool failed: {e}")

    logging.info(f"get_user_agenda_tool output: {out}")
    return json.dumps(out, ensure_ascii=False)

# (Optional) Re-enable if needed; keep the same tenant handling.
# def check_user_conflict_tool(input_str: str) -> str:
#     try:
#         logging.info(f"check_user_conflict_tool called with input: {input_str}")
#         args = _parse_payload(input_str)
#         result = check_user_conflict(
#             day=args["day"],
#             time_hhmm=args["time_hhmm"],
#             duration_min=int(args["duration_min"]),
#             user_id=args.get("user_id"),
#             tenant=args["tenant"],
#         )
#         out = {"success": True, "tenant": args["tenant"], **result}
#     except Exception as e:
#         out = {"success": False, "error": str(e)}
#         logging.error(f"check_user_conflict_tool failed: {e}")
#     logging.info(f"check_user_conflict_tool output: {out}")
#     return json.dumps(out, ensure_ascii=False)

# --------------------------------- Registry -----------------------------------

tools = [
    Tool(
        name="find_company",
        func=find_company_tool,
        description=(
            "Najdi firmu v CRM podle názvu (a/nebo města). "
            "Vstup: prostý text nebo JSON string "
            'např. {"query":"ACME Brno","tenant":"<TENANT>","top_k":5}. '
            "Vrací JSON se seznamem kandidátů včetně CRM ID. "
            "returns account id in items[i].id"
        ),
    ),
    Tool(
        name="find_contact",
        func=find_contact_tool,
        description=(
            "Najdi kontakt v CRM podle jména nebo e-mailu. "
            'Vstup: {"query":"Jan Pikna Invex","tenant":"<TENANT>","top_k":5}. '
            "Vrací JSON se seznamem kandidátů včetně CRM ID. "
            "returns contact id in items[i].id"
        ),
    ),
    Tool(
        name="list_contacts_by_company",
        func=list_contacts_by_company_tool,
        description=(
            "Vrať všechny kontakty dané firmy. "
            'Vstup: název firmy nebo {"account_id":"<GUID>","tenant":"<TENANT>"}. '
            "Výstup: JSON {company, contacts[]}."
        ),
    ),
    Tool(
        name="find_meetings",
        func=find_meetings_tool,
        description=(
            "Fuzzy search meetings by text. Use only if user asks about their planned meetings."
            'Vstup: {"query": string, "top_k"?: int, "date_from"?: "YYYY-MM-DD HH:MM", "date_to"?: "...", "tenant":"<TENANT>"}. '
            "Výstup: {meetings[]}."
        ),
    ),
    Tool(
        name="get_user_agenda",
        func=get_user_agenda_tool,
        description=(
            "Seznam schůzek pro daný den. "
            'Vstup: {"day":"YYYY-MM-DD","user_id"?: string, "tenant":"<TENANT>"}. '
            "Výstup: {agenda[]}."
        ),
    ),
    # Tool("check_user_conflict", func=check_user_conflict_tool, description="..."),
]
