# core/agents/module_data_extractor.py (refactored to single-shot, tool-free)

from core.services.llm import query_llm
from core.utils.chat import ChatSession
from core.agents.schemas import ModuleExtraction

availableModules = {"meetings", "tasks", "notes", "calls", "contacts", "accounts", "opportunities"}
# availableModules = {"meetings", "tasks", "notes", "calls"}

INSTRUCTION = f"""
You are an extractor. Return ONLY valid JSON matching this schema:

{{
"module": string | null,
"action": "create|update|delete|get|list|search" | null,
"parameters": {{
    "related_module": "company|contact|user|task|note|call|meeting|invoice" | null,
    "related_name": string | null
}}
}}

Rules:
- Use ONLY the allowed values above (lowercase).
- For "module", use the canonical CRM module name (e.g., Meetings, Contacts, Opportunities) or null.
- If uncertain, set fields to null.
- Do NOT include any extra fields or text outside JSON.
- Do NOT include "message_to_user".
- If company is mentioned, set related_module to "company".
"""

def _heuristics(p: str):
    p = p.lower()
    mh = "meetings" if ("schůzk" in p or "meeting" in p) else None
    if "úkol" in p: mh = mh or "tasks"
    if "poznámk" in p: mh = mh or "notes"
    if "kontakt" in p: mh = mh or "contacts"
    if any(w in p for w in ["hovor", "telefon", "call"]): mh = mh or "calls"
    if any(w in p for w in ["příležitost", "prilezitost", "opportunity"]): mh = mh or "opportunities"

    if any(w in p for w in ["vytvoř", "vytvor", "naplánuj", "založ"]): ah = "create"
    elif any(w in p for w in ["uprav", "změň", "přesuň"]): ah = "update"
    elif any(w in p for w in ["smaž", "zruš"]): ah = "delete"
    elif any(w in p for w in ["detail", "detaily", "získej", "najdi"]): ah = "get"
    elif any(w in p for w in ["ukaž", "vypiš", "seznam", "posledn", "nejbliž", "nadcházej", "jaké mám", "co mám"]): ah = "list"
    else: ah = None
    return mh, ah

def ModuleDataExtractor(prompt: str, chat_history: ChatSession = None) -> dict | None:
    # availableModules = {"meetings", "tasks", "notes", "calls", "contacts"}

    mh, ah = _heuristics(prompt)
    system_prompt = INSTRUCTION + f"\\nHINTS: module={mh or 'unknown'}, action={ah or 'unknown'}\\n"

    print(f"🛠️  ModuleDataExtractor prompt: {system_prompt}")

    # Single-shot: no prior history is necessary for extraction
    history = ChatSession(system_prompt=system_prompt, init=False)

    print("🛠️  ModuleDataExtractor querying LLM...")
    history.pretty_print()
    
    raw = query_llm(history, prompt, temperature=0, add_history=False, returnJson=True)

    print(f"🛠️  ModuleDataExtractor raw: {raw}")

    if isinstance(raw, dict) and "error" not in raw:
        try:
            parsed = ModuleExtraction(**raw)
        except Exception:
            return None
        data = parsed.dict()
        if not data.get("action"):
            data["action"] = "list"
        return data
    return None
