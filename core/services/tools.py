# core/services/tools.py (refactored)
# All LangChain Tools used by ReAct agents live here.
# Tools MUST:
#   - accept a simple str input (LangChain default),
#   - return a JSON string (so the agent can parse results easily).

from __future__ import annotations

import json
from typing import Any, Dict

from langchain.agents import Tool

from core.services.company_lookup import find_company_by_name
from core.services.contacts_lookup import find_contact_by_name_or_email
# from core.services.agenda import get_user_agenda  # optional agenda provider


# ------------------------------ Tool functions --------------------------------

def find_company_tool(name: str) -> str:
    """Input: company name as a plain string. Output: JSON string with company hits."""
    try:
        result = find_company_by_name(name)
        print(f"🤖 [CompanyLookupService] Found {len(result)} companies for '{name}'")
    except Exception as e:
        result = {"error": str(e)}
    return json.dumps(result, ensure_ascii=False)

def find_contact_tool(query: str) -> str:
    """Input: contact name or email (plain string). Output: JSON string with contact hits."""
    try:
        result = find_contact_by_name_or_email(query)
        print(f"🤖 [ContactsLookupService] Found {len(result)} contacts for '{query}'")
    except Exception as e:
        result = {"error": str(e)}
    return json.dumps(result, ensure_ascii=False)

# def get_agenda_tool(payload: str) -> str:
#     """
#     Input: JSON string -> {"user_id": "123", "date": "YYYY-MM-DD"}
#     Output: JSON string with agenda entries for that date.
#     """
#     try:
#         data = json.loads(payload) if payload else {}
#         user_id = data.get("user_id")
#         date = data.get("date")
#         result = get_user_agenda(user_id=user_id, date=date)
#     except Exception as e:
#         result = {"error": str(e)}
#     return json.dumps(result, ensure_ascii=False)


# --------------------------------- Registry -----------------------------------

tools = [
    Tool(
        name="find_company_by_name",
        func=find_company_tool,
        description=(
            "Najdi firmu v CRM dle názvu. Vstup je prostý řetězec s názvem firmy. "
            "Výstupem je JSON string s položkami firmy."
        ),
    ),
    Tool(
        name="find_contact_by_name_or_email",
        func=find_contact_tool,
        description=(
            "Najdi kontakt v CRM dle jména nebo e-mailu. Vstupem je prostý řetězec. "
            "Výstupem je JSON string s položkami kontaktu."
        ),
    ),
    # Tool(
    #     name="get_user_agenda",
    #     func=get_agenda_tool,
    #     description=(
    #         "Získej agendu uživatele pro daný den. Vstupem je JSON string: "
    #         '{"user_id":"...","date":"YYYY-MM-DD"}'
    #     ),
    # ),
]
