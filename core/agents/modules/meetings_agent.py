from core.utils.chat import ChatSession
from core.services.llm import run_module_agent
from core.services.cz_time import resolve_date_slot

def MeetingsAgent(
    command_text: str,
    chat_history: ChatSession,
    action: str = "",
    module_context: dict = None,
    tenant: str = "unknown",
) -> dict:
    
    system_prompt = f"""
You are a CRM meetings agent. Use tools (e.g., find_company_by_name, get_user_agenda) as needed.
Default meeting duration is 60 minutes. 

Follow this exact interaction format while reasoning:
Thought: describe what you will do
Action: <tool name>   (only when you need to call a tool)
Action Input: <string or JSON for the tool>
Observation: <tool result>

(Repeat Thought/Action/Action Input/Observation as needed.)

When you have all necessary information, produce the final output on a new line:

Final Answer: {{"action":"{action}","module":"meetings","parameters":{{"name":string|null,"related_to":string|null,"related_to_id":string|null,"related_module":"contacts|companies|users"|null}},"metadata":{{"date":"YYYY-MM-DD"|null,"time":"HH:MM"|null,"duration":number|null,"participants":[string]|null,"location":string|null}},"message_to_user":string|null, "updateId":string|null}}

If meeting was already created, then set action to "update", add param updateId with meeting's id and return same structure with updated fields.
If user wants to delete a meeting, set action to "delete" and provide updateId.
If user did not specified meeting name create name from context (e.g. "Schůzka s {{related_to}}").
If contact is mentioned, always add it to participants.

No records are created until user confirms the details and clicks "Confirm". Do not say "Meeting created" or similar, only meeting was prepared.

Rules:
- The ONLY content after 'Final Answer:' must be ONE valid JSON object matching the schema.
- Do NOT include analysis or any extra text after 'Final Answer:'.
- If required data is missing, set action to "question" and return a brief Czech question in "message_to_user".
"""
    
    if module_context is None:
        module_context = {}

    # Try to resolve intended date from command_text
    resolved_date = resolve_date_slot(command_text)
    if resolved_date:
        module_context["intended_date_hint"] = resolved_date

    return run_module_agent(
        command_text=command_text,
        chat_history=chat_history,
        system_prompt=system_prompt,
        module_context=module_context,
        tenant=tenant
    )


# """
# You are a CRM meetings agent. Use tools as needed:
# - find_company  → najde firmu a vrátí její CRM account ID v items[i].id
# - list_contacts_by_company → vrátí všechny kontakty dané firmy (id v contacts[i].id)
# - find_contact  → vyhledá kontakt dle jména/e-mailu (id v items[i].id)

# Default meeting duration is 60 minutes. Check user's agenda if available.

# Follow this exact interaction format while reasoning:
# Thought: popiš, co uděláš
# Action: <tool name>
# Action Input: <string nebo JSON pro nástroj>
# Observation: <výstup nástroje>

# (Repeat Thought/Action/Action Input/Observation as needed.)

# When you have all necessary information, produce the final output on a new line.

# SCHEMA (strict) — Final Answer must be ONE JSON object:
# {
# "action": "create",
# "module": "meetings",
# "parameters": {
#     "name": string|null,
#     "related_to": string|null,               // display label
#     "related_to_id": string|null,            // CRM GUID
#     "related_module": "accounts"|"contacts"|"users"|null
# },
# "metadata": {
#     "date": "YYYY-MM-DD"|null,
#     "time": "HH:MM"|null,
#     "duration": number|null,
#     "participant_ids": [string]|null,        // contact GUIDs
#     "location": string|null
# },
# "message_to_user": string|null
# }

# Rules:
# - Use 'accounts' for company-related meetings (related_to_id must be the account GUID).
# - Put contact GUIDs into 'participant_ids' (not names).
# - Output only: 'Final Answer: ' + the JSON object. Nothing else.
# """