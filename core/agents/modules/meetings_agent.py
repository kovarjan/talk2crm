import json
from core.services.llm import query_llm
from core.config import LLM_TEMPERATURE
from core.utils.json_validator import validate_json_command
from core.services.company_lookup import find_company_by_name
from core.utils.chat import ChatSession
from core.services.tools import tools
from langchain.agents import initialize_agent, AgentType
from langchain_ollama import OllamaLLM
from core.config import LLM_MODEL_NAME, LLM_TEMPERATURE

FUZZY_MATCH_THRESHOLD = 0.6

llm = OllamaLLM(model=LLM_MODEL_NAME, temperature=LLM_TEMPERATURE)

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

def MeetingsAgent(
    command_text: str,
    chat_history: ChatSession,
    action: str = "",
    module_context: dict = None,
) -> dict:
    print(f"🤖 [MeetingsAgent] Command Text: {command_text}")

    if not chat_history:
        chat_history = ChatSession()

    # Add user command and prompt to history
    chat_history.add_user(command_text)

    # Prepare context string if module_context is provided
    context_str = ""
    if module_context:
        context_items = []
        for key, value in module_context.items():
            context_items.append(f"{key}: {value}")
        context_str = "CONTEXT:\n" + "\n".join(context_items) + "\n"

        chat_history.set_module("meetings")
        chat_history.set_action(action)

    # Schema and prompt guidance
    schema_prompt = f"""
{context_str}
You are a CRM meetings agent. Your job is to schedule or retrieve meetings.
You can use tools like `find_company_by_name` or `get_user_agenda` if needed.
Default meeting duration is 1 hour. Check user's agenda for conflicts.

Your final output must follow this JSON schema:

{{
    "action": "{action}",
    "module": "meetings",
    "parameters": {{
        "name": "<meeting_name>",
        "related_to": "<related_to name>",
        "related_to_id": "<related_to_id>",
        "related_module": "<companies, contacts, users>",
        "...": "..."
    }},
    "metadata": {{
        "date": "<date>",
        "time": "<time>",
        "duration": "<minutes>",
        "participants": "...",
        "location": "..."
    }},
    "message_to_user": "<clear message to user in Czech>"
}}

If required data is missing, respond with:
{{
    "action": "question",
    "message_to_user": "<ask for missing info in Czech>"
}}

Always use JSON format only. No comments or extra text.
"""
    chat_history.add_system(schema_prompt)

    # Let the agent process the full instruction
    langchain_messages = chat_history.to_langchain_messages()
    try:
        output_text = agent.invoke(langchain_messages)
    except Exception as e:
        print(f"❌ Agent failed: {e}")
        return {
            "action": "error",
            "message_to_user": "Omlouvám se, došlo k chybě při zpracování požadavku. Můžete to prosím zkusit znovu?"
        }

    # Parse the output as JSON (LLM should return JSON as per your prompt)
    try:
        print(f"🤖 [MeetingsAgent] Raw Output: {output_text}")
        # Use the .output property if present (LangChain agent returns a dict)
        if isinstance(output_text, dict) and "output" in output_text:
            cleaned = output_text["output"].strip()
        else:
            cleaned = str(output_text).strip()

        print(f"🤖 [MeetingsAgent] Cleaned Output: {cleaned}")
        result = json.loads(cleaned)
    except Exception as e:
        print(f"❌ JSON parse error: {e}")
        return {
            "action": "question",
            "message_to_user": "Omlouvám se, došlo k chybě při zpracování požadavku. Můžete to prosím zkusit znovu?"
        }

    # Optional: Validate JSON schema
    validation = validate_json_command(result)
    if not validation["is_valid"]:
        print(f"❌ Validation error: {validation['errors']}")
        return {
            "action": "error",
            "message_to_user": "Omlouvám se, došlo k chybě při zpracování požadavku. Můžete prosím specifikovat požadavek přesněji?"
        }
    
    # Add the response to chat history
    chat_history.add_llm_response(result)

    return result
