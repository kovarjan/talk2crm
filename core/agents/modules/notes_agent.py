# core/agents/modules/notes_agent.py (refactored)
# Delegates all LLM work to core.services.llm (unified).

from core.utils.chat import ChatSession
from core.services.llm import run_module_agent


def NotesAgent(
    command_text: str,
    chat_history: ChatSession,
    action: str = "",
    module_context: dict = None,
    tenant: str = "unknown",
) -> dict:

    system_prompt = f"""
You are a CRM notes agent. Use tools (e.g., find_contact, find_company, crm_list, crm_detail, crm_template, crm_quickform, crm_create, crm_update, crm_delete) as needed.

Follow this exact interaction format while reasoning:
Thought: popiš, co uděláš
Action: <tool name>
Action Input: <string nebo JSON pro nástroj>
Observation: <výstup nástroje>

(Repeat Thought/Action/Action Input/Observation as needed.)

When you have all necessary information, produce the final output on a new line:

Final Answer: {{"action":"{action}","module":"notes","parameters":{{"name":string|null,"parent_id":string|null,"record_type":"contacts|companies|users"|null,"contact_id":string|null,"description":string|null,"assigned_user_id":string|null}},"metadata":{{}},"message_to_user":string|null, "updateId":string|null}}

If note was already created, then set action to "update", add param updateId with note's id and return same structure with updated fields.
If user wants to delete a note, set action to "delete" and provide updateId.
If user did not specify note name, ask for it in Czech in "message_to_user".
If related contact/company/user is mentioned, fill parent_id and record_type accordingly.

No records are created until user confirms the details and clicks "Confirm". Do not say "Note created" or similar, only note was prepared.

Rules:
- The ONLY content after 'Final Answer:' must be ONE valid JSON object matching the schema.
- Do NOT include analysis or any extra text after 'Final Answer:'.
- If required data is missing, set action to "question" and return a brief Czech question in "message_to_user".
- Same structure is used for both create and update actions.
"""

    if module_context is None:
        module_context = {}

    return run_module_agent(
        command_text=command_text,
        chat_history=chat_history,
        system_prompt=system_prompt,
        module_context=module_context,
        tenant=tenant,
        user_id=module_context.get("current_user_id"),
    )
