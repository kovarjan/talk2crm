# core/services/tools.py
# All LangChain Tools used by ReAct agents live here.
# Tools MUST:
#   - accept a simple str input (LangChain default),
#   - return a JSON string (so the agent can parse results easily).

from __future__ import annotations

import json
import os
from typing import Any, Dict

from langchain.agents import Tool

# Updated imports: these use the new vector store in var/vector and support tenant
from core.services.company_lookup import find_company_by_name_or_city
from core.services.contacts_lookup import find_contact_by_query
from core.services.company_contacts import list_contacts_for_account, list_contacts_for_company_name

DEFAULT_TENANT = os.getenv("DEFAULT_TENANT", "ai-local")
DEFAULT_TOP_K = int(os.getenv("VECTOR_TOP_K", "5"))


# def _parse_payload(payload: str) -> Dict[str, Any]:
#     """
#     Accepts either a plain string (query) or a JSON string.
#     Returns dict with keys: query (str), tenant (str), top_k (int).
#     """
#     if not payload:
#         raise ValueError("Empty input. Provide a company/contact name or a JSON string.")
#     try:
#         obj = json.loads(payload)
#         if isinstance(obj, dict):
#             query = obj.get("query") or obj.get("name") or obj.get("q")
#             if not query:
#                 raise ValueError("JSON must include 'query' field.")
#             tenant = obj.get("tenant", DEFAULT_TENANT)
#             top_k = int(obj.get("top_k", DEFAULT_TOP_K))
#             return {"query": query, "tenant": tenant, "top_k": top_k}
#         # If the JSON parsed but is not a dict (e.g., ["foo"]), treat as string below.
#     except Exception:
#         # Not JSON → treat as plain query string
#         pass
#     return {"query": payload.strip(), "tenant": DEFAULT_TENANT, "top_k": DEFAULT_TOP_K}


def _parse_payload(payload: str) -> Dict[str, Any]:
    if not payload:
        raise ValueError("Empty input")
    try:
        obj = json.loads(payload)
        if isinstance(obj, dict):
            q = obj.get("query") or obj.get("name")
            return {
                "query": q if q else None,
                "tenant": obj.get("tenant", DEFAULT_TENANT),
                "top_k": int(obj.get("top_k", DEFAULT_TOP_K)),
                "account_id": obj.get("account_id"),
            }
    except Exception:
        pass
    return {"query": payload.strip(), "tenant": os.getenv("DEFAULT_TENANT", "ai-local"), "top_k": 5, "account_id": None}

def list_contacts_by_company_tool(input_str: str) -> str:
    """
    Input (either):
      - plain string: "UNISTAV CONSTRUCTION a.s."
      - JSON: {"query":"UNISTAV CONSTRUCTION a.s.","tenant":"ai-local"}
      - JSON: {"account_id":"<GUID>","tenant":"ai-local"}

    Output JSON:
      {"success":true,"company":{...},"contacts":[{"id":"...","name":"...","email":"...","phone":"..."}, ...]}
    """
    import json
    try:
        args = _parse_payload(input_str)
        tenant = args["tenant"]
        if args.get("account_id"):
            contacts = list_contacts_for_account(tenant, args["account_id"])
            out = {"success": True, "company": {"id": args["account_id"]}, "contacts": contacts}
        else:
            res = list_contacts_for_company_name(args["query"], tenant=tenant)
            out = {"success": True, **res}
    except Exception as e:
        out = {"success": False, "error": str(e)}
    return json.dumps(out, ensure_ascii=False)


# ------------------------------ Tool functions --------------------------------

def find_company_tool(input_str: str) -> str:
    """
    Input (either):
      - plain string: "ACME Brno"
      - JSON string: {"query":"ACME Brno","tenant":"ai-local","top_k":5}
    Output:
      JSON string: {"success":true,"items":[{"id": "...", "name":"...", "city":"...", "score": 0.78, "raw": {...}}, ...]}
    """
    try:
        args = _parse_payload(input_str)
        hits = find_company_by_name_or_city(args["query"], tenant=args["tenant"], top_k=args["top_k"])
        out = {"success": True, "tenant": args["tenant"], "query": args["query"], "items": hits}
    except Exception as e:
        out = {"success": False, "error": str(e)}
    return json.dumps(out, ensure_ascii=False)


def find_contact_tool(input_str: str) -> str:
    """
    Input (either):
      - plain string: "Jan Pikna Invex" or "jan@invex.cz"
      - JSON string: {"query":"Jan Pikna Invex","tenant":"ai-local","top_k":5}
    Output:
      JSON string: {"success":true,"items":[{"id":"...","name":"Jan Pikna, INVEX","email":"...","phone":"...","account_id":"...","score":0.73,"raw":{...}}, ...]}
    """
    try:
        args = _parse_payload(input_str)
        hits = find_contact_by_query(args["query"], tenant=args["tenant"], top_k=args["top_k"])
        out = {"success": True, "tenant": args["tenant"], "query": args["query"], "items": hits}
    except Exception as e:
        out = {"success": False, "error": str(e)}
    return json.dumps(out, ensure_ascii=False)


# --------------------------------- Registry -----------------------------------

tools = [
    Tool(
        name="find_company",
        func=find_company_tool,
        description=(
            "Najdi firmu v CRM podle názvu (a/nebo města). "
            "Vstup může být prostý text (např. 'ACME Brno'), nebo JSON string "
            'např. {"query":"ACME Brno","tenant":"ai-local","top_k":5}. '
            "Vrací JSON se seznamem kandidátů včetně CRM ID."
            "returns account id in items[i].id"
        ),
    ),
    Tool(
        name="find_contact",
        func=find_contact_tool,
        description=(
            "Najdi kontakt v CRM podle jména nebo e-mailu. "
            "Vstup je buď prostý text (např. 'Jan Pikna Invex'), nebo JSON string "
            'např. {"query":"jan@invex.cz","tenant":"ai-local","top_k":5}. '
            "Vrací JSON se seznamem kandidátů včetně CRM ID."
            "returns contact id in items[i].id"
        ),
    ),
    Tool(
        name="list_contacts_by_company",
        func=list_contacts_by_company_tool,
        description=(
            "Vrať všechny kontakty patřící do dané firmy (podle CRM). "
            "Vstup může být název firmy (např. 'UNISTAV CONSTRUCTION a.s.') "
            "nebo JSON s account_id: "
            '{"account_id":"<GUID>","tenant":"ai-local"}.\n'
            "Výstup je JSON se seznamem kontaktů {id, name, email, phone}."
            "returns contact ids in contacts[i].id"
        ),
    ),
]
