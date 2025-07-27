import json
from core.services.llm import query_llm
from core.config import LLM_TEMPERATURE
from core.utils.json_validator import validate_json_command
from core.services.company_lookup import find_company_by_name
from core.utils.chat import ChatSession

FUZZY_MATCH_THRESHOLD = 0.6


def MeetingsAgent(
    command_text: str,
    chat_history: ChatSession,
    action: str = "",
    parameters: list = None,
) -> dict:
    """
    Process the command text and generate a JSON response for meeting-related tasks.

    Args:
        command_text (str): The command text to process.
        chat_history (list, optional): The chat history for context. Defaults to None.

    Returns:
        dict: A JSON response containing the action, parameters, and metadata.
    """

    print(f"🤖 [MeetingsAgent] Command Text: {command_text}")

    matched_company = ""
    extra_context = ""
    resolved_company = None


    if parameters.get("related_module") == "company" and parameters.get("related_name"):
        print(f"🔎 Matched company from command: {matched_company}")
        matched_company = parameters.get("related_name")

        if matched_company:
            print(f"🔎 Trying to resolve company name: {matched_company}")
            results = find_company_by_name(matched_company)
            print(f"🔎 Found {len(results)} results for '{matched_company}'")
            print(f"🔎 Results: {results}")
            if results and results[0]["score"] > FUZZY_MATCH_THRESHOLD:
                resolved_company = results[0]
                print(f"✅ Resolved to: {resolved_company}")
            else:
                print("⚠️ Could not confidently resolve company name.")

        if resolved_company:
            extra_context += f'\nMatched company from CRM:\n- Name: {resolved_company["name"]}\n- ID: {resolved_company["id"]}\n'

    # Inject the resolved company context into the chat history
    if resolved_company:
        chat_history.inject_context(
            label="Company",
            name=resolved_company["name"],
            id=resolved_company["id"],
        )

    # Add user message to chat history
    chat_history.add_user(command_text)

    # Define the prompt template for generating the JSON command
    prompt_template = f"""
You are an meetings agent your task is to schedule or retrieve meetings for user of CRM system.
Given users command and conversation history, generate a JSON action according to the schema provided below.

Schema:
{{
    "action": "{action}",
    "module": "meetings",
    "parameters": {{
        "name": "<meeting_name> [required]",
        "related_to": "<related_to meeting with (company, contact, etc.) name from context> [required]",
        "related_to_id": "<related_to_id related module ID from context> [required]",
        "related_module": "<related_module companies, contacts, users, etc.> [required]",
        "<param_name>": "<param_value>"
    }},
    "metadata": {{
        "date": "<date> [required]",
        "time": "<time> [required]",
        "duration": "<duration minutes if specified>",
        "participants": "<participants if specified>",
        "location": "<location if specified>",
    }},
    "message_to_user": "<message to user with confirmation or error message> [required]"
}}

If any of the required fields are missing, please add clarification schema JSON with message to ask the user for more details and action like question, error, or confirmation.

Clarification schema:
{{
    ...
    "message_to_user": "<clarification message to user>",
    "action": "<action required like question, error, or confirmation>",
}}

Please ensure the output is a valid JSON object with no extra text. If additional information or confirmation is needed,
add clarification schema with message and action "question". 
Always include message_to_user field with a clear message for the user in czech language, every action must be confirmed by user.
    """

    # Add schema as another system message to the chat history
    chat_history.add_system(prompt_template)
    
    with open("logs/llm_response.log", "a", encoding="utf-8") as log_file:
        log_file.write(f"MeetingsAgent\n")

    # Generate JSON from the command text
    json_response = query_llm(chat_history=chat_history, temperature=LLM_TEMPERATURE)
    print(f"🤖 [LLM] Generated JSON Command: {json.dumps(json_response, indent=4)}")

    validation = validate_json_command(json_response)
    if not validation["is_valid"]:
        print(
            f"❌ [Validation] Errors: {validation.get('errors', 'Invalid JSON command generated.')}"
        )

    # Validate JSON output here
    if "error" in json_response:
        # Optionally, trigger a clarification/confirmation step
        json_response = {
            "action": "question",
            "message_to_user": "Omlouvám, nepodařilo se mi vyhodnotit tento úkol. Můžete to prosím zopakovat?"
        }

    return json_response
