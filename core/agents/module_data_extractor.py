# core/agents/module_extractor.py
import json
from core.services.llm import query_llm
from core.utils.chat import ChatSession
from core.agents.schemas import ModuleExtraction
# from .lexicon import CONTACT_NAMES, COMPANY_NAMES  # preload at app start
# from rapidfuzz import process, fuzz


INSTRUCTION = """
You are an extractor. Return ONLY valid JSON matching this schema:

{
"module": "meetings|tasks|notes|calls",
"action": "create|update|delete|get",
"parameters": {
    "related_module": "company|contact|user|task|note|call|meeting|invoice" | null,
    "related_name": string | null
}
}

Rules:
- Use ONLY the allowed values above (lowercase).
- If uncertain, set fields to null.
- Do NOT include any extra fields or text outside JSON.
- Do NOT include "message_to_user".

Examples:
User: "Vytvoř schůzku s Janem Piknou z firmy INVEX na pátek 10:45."
Return:
{"module":"meetings","action":"create","parameters":{"related_module":"contact","related_name":"Jan Pikna"}}

User: "Zruš úkol u firmy IMOS."
Return:
{"module":"tasks","action":"delete","parameters":{"related_module":"company","related_name":"IMOS"}}
"""


# def _snap(name, pool):
#     if not name: return name
#     m = process.extractOne(name, pool, scorer=fuzz.WRatio, score_cutoff=87)
#     return m[0] if m else name

def ModuleDataExtractor(prompt: str, chat_history: ChatSession = None) -> dict | None:
    availableModules = ["meetings", "tasks", "notes", "calls"]

    # Build system prompt with light hints
    def heuristics(p):
        p = p.lower()
        mh = "meetings" if "schůzk" in p or "meeting" in p else None
        if "úkol" in p: mh = mh or "tasks"
        if "poznámk" in p: mh = mh or "notes"
        if any(w in p for w in ["hovor", "telefon", "call"]): mh = mh or "calls"

        if any(w in p for w in ["vytvoř", "vytvor", "naplánuj", "založ"]): ah = "create"
        elif any(w in p for w in ["uprav", "změň", "přesuň"]): ah = "update"
        elif any(w in p for w in ["smaž", "zruš"]): ah = "delete"
        elif any(w in p for w in ["ukaž", "získej", "najdi", "vypiš"]): ah = "get"
        else: ah = None
        return mh, ah

    mh, ah = heuristics(prompt)
    system_prompt = INSTRUCTION + f"\nHINTS: module={mh or 'unknown'}, action={ah or 'unknown'}\n"

    print(f"🤖 [ModuleDataExtractor] System Prompt: {system_prompt}")

    history = ChatSession(system_prompt=system_prompt)
    if chat_history:
        history.load_history(chat_history.get_user_assistant_messages())

    raw = query_llm(history, prompt, temperature=0)

    print(f"🤖 [ModuleDataExtractor] Raw Output: {raw}")

    # extract json
    # start, end = raw.find("{"), raw.rfind("}")
    # data = json.loads(raw[start:end+1]) if start!=-1 and end!=-1 else {}
    data = raw

    try:
        parsed = ModuleExtraction(**data)
    except Exception:
        # one retry with stricter reminder
        retry = query_llm(history, f"{system_prompt}\nReturn ONLY valid JSON for: {prompt}", temperature=0)
        s2, e2 = retry.find("{"), retry.rfind("}")
        data = json.loads(retry[s2:e2+1]) if s2!=-1 and e2!=-1 else {}
        parsed = ModuleExtraction(**data)

    # snap related_name to known entities when appropriate
    # if parsed.parameters and parsed.parameters.related_name:
    #     if parsed.parameters.related_module == "contact":
    #         parsed.parameters.related_name = _snap(parsed.parameters.related_name, CONTACT_NAMES)
    #     elif parsed.parameters.related_module == "company":
    #         parsed.parameters.related_name = _snap(parsed.parameters.related_name, COMPANY_NAMES)

    # final guardrails: null-out unsupported values
    if parsed.module not in availableModules or not parsed.action:
        return None

    return json.loads(parsed.json())
