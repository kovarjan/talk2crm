"""
command_pipeline.py

This module orchestrates the end-to-end workflow from receiving a voice command to executing a CRM action.
It leverages agents for Speech-to-Text (STT), LLM processing, JSON validation, and the CRM service.
"""
import json
from core.services.stt import transcribe_audio
from core.services.llm import query_llm
from core.utils.json_validator import validate_json_command
from core.agents.module_data_extractor import ModuleDataExtractor
from core.agents.modules.meetings_agent import MeetingsAgent
# from core.services.crm_service import process_crm_command

def process_voice_command(audio_path: str, chat_history: list = []) -> dict:
    """
    Processes a voice command from an audio file and executes the corresponding CRM action.

    Args:
        audio_path (str): The file path of the audio input.
        chat_history (list, optional): Previous conversation context as a list of dicts
                                       with keys "role" and "content".

    Returns:
        dict: The final response from the CRM service or an error message if encountered.
    """
    # NOTE: Audio transcription is disabled out for now.
    try:
        # Convert audio to text using the STT agent.
        command_text = transcribe_audio(audio_path)
        print(f"🎙️ [STT] Transcribed Text: {command_text}")
    except Exception as stt_error:
        return {"error": f"🎙️ STT failed: {str(stt_error)}"}

    # NOTE: Manual override for testing
    command_text = "Vytvoř schůzku s Pavlem Novotným ve 12 hodin v Brně."

    print("\n🗣️ User said (manual override):", command_text)


    # Extract module data
    module_data = ModuleDataExtractor(command_text)
    if not "error" in module_data:
        # If module data extraction is successful, proceed with the command text.
        print(f"🛠️ [ModuleDataExtractor]", module_data)
        chat_history.append({
            "role": "assistant",
            "content": f"[ModuleDataExtractor] Module: '{module_data.get('module')}', Action: '{module_data.get('action')}'"
        })

        print(chat_history)
    else:
        print(f"🛠️ [ModuleDataExtractor] Error: {module_data['error']}")
        
    # based on module select correct module_agent default to query_llm
    if module_data.get("module") == "meetings":
        # meetings_agent = MeetingsAgent(command_text, chat_history=chat_history, action=module_data.get("action"))
        # response = meetings_agent
        response = MeetingsAgent(command_text, chat_history=chat_history, action=module_data.get("action"))
    elif module_data.get("module") == "tasks":
        print("🛠️ [TasksAgent] Not implemented yet.")
        return {"error": "Tasks module not implemented yet."}
    else:
        # Generate the JSON command using the LLM agent, including any chat history if available.
        response = query_llm(command_text, chat_history=chat_history)
        print(f"🤖 [LLM] Generated JSON Command: {response}")

    if "error" in response:
        # Optionally, trigger a clarification/confirmation step
        response = {"clarification": {
            "question": "Omlouvám, nepodařilo se mi vyhodnotit tento úkol. Můžete to prosím zopakovat?"
        }}

    # Validate the generated JSON command structure.
    validation = validate_json_command(response)
    if not validation["is_valid"]:
        # If validation fails, return an error or trigger clarification (implementation-dependent).
        return {
            "error": "Invalid JSON command generated.",
            "details": validation.get("errors", "Omlouvám, nepodařilo se mi vyhodnotit tento úkol. Můžete to prosím zopakovat?")
        }

    # If the command has property clarification.question rerun it
    if "clarification" in response:

        if "question" in response["clarification"]:
            # If the response contains a clarification question, ask the user for more details.
            print("\n❓ Clarification needed:", response["clarification"]["question"])

        # Handle clarification logic here
        # For example, you can ask the user for more details
        user_input = input("\n💡 Please provide more details: ")

        # user_corrected_text = "original message: " + text + "\n\nyour response: " + response + "\n\nusers clarification: " + user_input
        user_corrected_text = [{
            "role": "user",
            "content": command_text
        }, {
            "role": "assistant",
            "content": json.dumps(response, indent=4)
        }]

        response = process_voice_command(user_input, user_corrected_text)
        print("\n🤖 Assistant Response:\n", response)

    # Process the command through the CRM service.
    # crm_response = process_crm_command(response)
    # return crm_response

    return response
