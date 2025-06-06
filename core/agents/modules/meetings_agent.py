import json
from langchain.prompts import PromptTemplate
from core.services.llm import query_llm
from core.config import LLM_TEMPERATURE
from core.utils.json_validator import validate_json_command


def MeetingsAgent(command_text: str, chat_history: list = None, action: str = '') -> dict:
    """
    Process the command text and generate a JSON response for meeting-related tasks.
    
    Args:
        command_text (str): The command text to process.
        chat_history (list, optional): The chat history for context. Defaults to None.
    
    Returns:
        dict: A JSON response containing the action, parameters, and metadata.
    """

    print(f"🤖 [MeetingsAgent] Command Text: {command_text}")

    prompt_template = """
    Given the following voice command and conversation history, generate a JSON action according to the schema provided below.

    Command: "{command_text}"

    Schema:
    {{
        "action": "{action}",
        "module": "meetings",
        "parameters": {{
            "name": "<meeting_name> [required]",
            "related_to": "<related_to meeting with (company, contact, etc.)> [required]",
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

    Please ensure the output is a valid JSON object with no extra text. If additional information or confirmation is needed,
    include a field such as "clarification" or "question" instead of the action.
    If any of the required fields are missing, please return an clarification message.

    Clarification schema:
    {{
        "clarification": "<clarification message>",
        "action": "<action if applicable>",
    }}
    """
    
    # Prepare the prompt using LangChain's template.
    prompt = PromptTemplate(template=prompt_template, input_variables=["command_text", "action"])
    final_prompt = prompt.format(command_text=command_text, action=action)

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
        json_response = {"clarification": {
            "question": "Omlouvám, nepodařilo se mi vyhodnotit tento úkol. Můžete to prosím zopakovat?"
        }}

    return json_response