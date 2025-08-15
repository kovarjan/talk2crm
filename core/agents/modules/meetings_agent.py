# core/agents/modules/meetings_agent.py (refactored)
# Delegates all LLM work to core.services.llm (unified).

from core.utils.chat import ChatSession
from core.services.llm import run_module_agent

def MeetingsAgent(
    command_text: str,
    chat_history: ChatSession,
    action: str = "",
    module_context: dict = None,
) -> dict:
    
    system_prompt = f"""
You are a CRM meetings agent. Use tools (e.g., find_company_by_name, get_user_agenda) as needed.
Default meeting duration is 60 minutes. Check user's agenda for conflicts.

Follow this exact interaction format while reasoning:
Thought: describe what you will do
Action: <tool name>   (only when you need to call a tool)
Action Input: <string or JSON for the tool>
Observation: <tool result>

(Repeat Thought/Action/Action Input/Observation as needed.)

When you have all necessary information, produce the final output on a new line:

Final Answer: {{"action":"{action}","module":"meetings","parameters":{{"name":string|null,"related_to":string|null,"related_to_id":string|null,"related_module":"contacts|companies|users"|null}},"metadata":{{"date":"YYYY-MM-DD"|null,"time":"HH:MM"|null,"duration":number|null,"participants":[string]|null,"location":string|null}},"message_to_user":string|null}}

Rules:
- The ONLY content after 'Final Answer:' must be ONE valid JSON object matching the schema.
- Do NOT include analysis or any extra text after 'Final Answer:'.
- If required data is missing, set action to "question" and return a brief Czech question in "message_to_user".
"""

    return run_module_agent(
        command_text=command_text,
        chat_history=chat_history,
        system_prompt=system_prompt,
        module_context=module_context,
    )
