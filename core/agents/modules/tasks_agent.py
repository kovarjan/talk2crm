from core.utils.chat import ChatSession
from core.services.llm import run_module_agent
from core.services.cz_time import resolve_date_slot


def TasksAgent(
    command_text: str,
    chat_history: ChatSession,
    action: str = "",
    module_context: dict = None,
    tenant: str = "unknown",
) -> dict:

    system_prompt = f"""
You are a CRM tasks agent. Use tools (e.g., find_contact, find_company, list_contacts_by_company) as needed.

Follow this exact interaction format while reasoning:
Thought: popiš, co uděláš
Action: <tool name>
Action Input: <string nebo JSON pro nástroj>
Observation: <výstup nástroje>

(Repeat Thought/Action/Action Input/Observation as needed.)

When you have all necessary information, produce the final output on a new line:

Final Answer: {{"action":"{action}","module":"tasks","parameters":{{"name":string|null,"status":string|null,"date_start":"YYYY-MM-DD"|null,"date_due":"YYYY-MM-DD"|null,"parent_id":string|null,"parent_type":"contacts|companies|users"|null,"contact_id":string|null,"all_day":boolean|null,"assigned_user_id":string|null,"description":string|null}},"metadata":{{}},"message_to_user":string|null}}

If task was already created, then set action to "update", add param updateId with task's id and return same structure with updated fields.
If user wants to delete a task, set action to "delete" and provide updateId.
If user did not specify task name, ask for it in Czech in "message_to_user".
If related contact/company/user is mentioned, fill parent_id and parent_type accordingly.
Status must be one of: "Not Started", "In Progress", "Completed", "Pending Input", "Deferred"

No records are created until user confirms the details and clicks "Confirm". Do not say "Task created" or similar, only task was prepared.

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
        tenant=tenant,
    )
