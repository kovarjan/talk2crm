from core.utils.chat import ChatSession
from core.services.llm import run_module_agent


def QueryAgent(
    command_text: str,
    chat_history: ChatSession,
    action: str = "list",
    module: str = "",
    module_context: dict = None,
    tenant: str = "unknown",
) -> dict:
    system_prompt = f"""
You are a CRM query agent. Use CRM REST tools (crm_list, crm_detail, crm_template, crm_quickform) to answer user questions.
Always call at least one crm_* tool to fetch real data before answering.

When you have the data, produce the final output on a new line:

Final Answer: {{"action":"{action}","module":"{module}","parameters":{{"filter":object|null,"order":object|null,"limit":number|null,"offset":number|null,"record":string|null}} ,"message_to_user":string, "crm_data":object|null, "already_fetched":true}}

Rules:
- For list/search questions, set action to "list" and call crm_list.
- For detail questions about one record, set action to "get" and call crm_detail.
- If user asks "poslední / nejbližší / nadcházející", use order by date_start desc for meetings; otherwise use date_modified desc.
- Default limit is 5 unless user specifies otherwise.
- Put the tool result into crm_data (raw JSON), and summarize it in message_to_user.
- If no records found, message_to_user should say so in Czech.
- The ONLY content after 'Final Answer:' must be ONE valid JSON object.
"""

    if module_context is None:
        module_context = {}

    module_context.setdefault("query_module", module)

    return run_module_agent(
        command_text=command_text,
        chat_history=chat_history,
        system_prompt=system_prompt,
        module_context=module_context,
        tenant=tenant,
        user_id=module_context.get("current_user_id"),
    )
