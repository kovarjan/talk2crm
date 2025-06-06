import json
import re
from langchain.prompts import PromptTemplate
from core.services.llm import query_llm
from core.config import LLM_TEMPERATURE
from core.utils.json_validator import validate_json_command
from core.services.company_lookup import find_company_by_name

FUZZY_MATCH_THRESHOLD = 0.62

def MeetingsAgent(command_text: str, chat_history: list = None, action: str = '', parameters: list = None) -> dict:
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

    if parameters.get('related_module') == "company" and parameters.get("related_name"):
        print(f"🔎 Matched company from command: {matched_company}")
        matched_company = parameters.get("related_name")
     
        resolved_company = None
        if matched_company:
            print(f"🔎 Trying to resolve company name: {matched_company}")
            results = find_company_by_name(matched_company)
            print(f"🔎 Found {len(results)} results for '{matched_company}'")
            print(f"🔎 Results: {results}")
            if results and results[0]['score'] > FUZZY_MATCH_THRESHOLD:
                resolved_company = results[0]
                print(f"✅ Resolved to: {resolved_company}")
            else:
                print("⚠️ Could not confidently resolve company name.")

        if resolved_company:
            extra_context += f'\nMatched company from CRM:\n- Name: {resolved_company["name"]}\n- ID: {resolved_company["id"]}\n'


    prompt_template = """
    Given the following voice command and conversation history, generate a JSON action according to the schema provided below.

    Command: "{command_text}"

    Context:
    {extra_context}

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
        }}
    }}

    If any of the required fields are missing, please add clarification schema JSON with message to ask the user for more details and action like question, error, or confirmation.

    Clarification schema:
    {{
        ...
        "clarification": "<clarification message to user>",
        "action": "<action required like question, error, or confirmation>",
    }}

    Please ensure the output is a valid JSON object with no extra text. If additional information or confirmation is needed,
    add clarification schema with message and action "question". Always respond in czech language.
    """
    
    # Prepare the prompt using LangChain's template.
    prompt = PromptTemplate(template=prompt_template, input_variables=["command_text", "action", "extra_context"])
    final_prompt = prompt.format(command_text=command_text, action=action, extra_context=extra_context)

    print(f"🛠️ [Prompt] Final Prompt: {final_prompt}")

    systemPrompt = "You are an meetings agent your task is to schedule or retrieve meetings for user of CRM system."

    # Generate JSON from the command text
    json_response = query_llm(final_prompt, chat_history=chat_history, temperature=LLM_TEMPERATURE, isTemplate=True, systemPrompt=systemPrompt)
    print(f"🤖 [LLM] Generated JSON Command: {json.dumps(json_response, indent=4)}")


    validation = validate_json_command(json_response)
    if not validation["is_valid"]:
        print(f"❌ [Validation] Errors: {validation.get('errors', 'Invalid JSON command generated.')}")


    # Validate JSON output here
    if "error" in json_response:
        # Optionally, trigger a clarification/confirmation step
        # json_response = {"clarification": "The command was unclear. Could you repeat with contact details?"}
        json_response = {
            "clarification": "Omlouvám, nepodařilo se mi vyhodnotit tento úkol. Můžete to prosím zopakovat?"
        }

    return json_response
