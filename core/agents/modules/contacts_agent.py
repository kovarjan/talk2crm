# core/agents/modules/meetings_agent.py (refactored)
# Delegates all LLM work to core.services.llm (unified).

from core.utils.chat import ChatSession
from core.services.llm import run_module_agent
from core.services.cz_time import resolve_date_slot

def ContactsAgent(
    command_text: str,
    chat_history: ChatSession,
    action: str = "",
    module_context: dict = None,
    tenant: str = "unknown",
) -> dict:
    
    system_prompt = f"""
You are a CRM contacts agent. Use tools (e.g., find_company, list_contacts_by_company, find_contact) as needed.

Follow this exact interaction format while reasoning:
Thought: popiš, co uděláš
Action: <tool name>
Action Input: <string nebo JSON pro nástroj>
Observation: <výstup nástroje>

(Repeat Thought/Action/Action Input/Observation as needed.)

When you have all necessary information, produce the final output on a new line:

Final Answer: {{"action":"{action}","module":"contacts","parameters":{{"first_name":string|null,"last_name":string|null,"full_name":string|null,"email1":string|null,"phone_mobile":string|null, "department":string|null, "title":string|null, "description":string|null, "account_id":string|null}},"metadata":{{"notes":string|null}},"message_to_user":string|null, "updateId":string|null}}

If contact was already created, then set action to "update", add param updateId with contact's id and return same structure with updated fields.
If user wants to delete a contact, set action to "delete" and provide updateId.
If user did not specify contact name, ask for it in Czech in "message_to_user".
If company is mentioned, always link contact to company using "account_id".

No records are created until user confirms the details and clicks "Confirm". Do not say "Contact created" or similar, only contact was prepared.

Rules:
- The ONLY content after 'Final Answer:' must be ONE valid JSON object matching the schema.
- Do NOT include analysis or any extra text after 'Final Answer:'.
- If required data is missing, set action to "question" and return a brief Czech question in "message_to_user".
- Same structure is used for both create and update actions.
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
