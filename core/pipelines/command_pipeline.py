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
import time
from core.utils.chat import ChatSession
from core.config import SHOW_TIMING

# from core.services.crm_service import process_crm_command

def process_voice_command(audio_path: str, chat_history: ChatSession = ChatSession(), input_text: str = None) -> dict:
    """
    Processes a voice command from an audio file and executes the corresponding CRM action.

    Args:
        audio_path (str): The file path of the audio input.
        chat_history (list, optional): Previous conversation context as a list of dicts
                                       with keys "role" and "content".

    Returns:
        dict: The final response from the CRM service or an error message if encountered.
    """
    processTime = time.time()
    command_text = input_text

    if audio_path:
        try:
            # Convert audio to text using the STT agent.
            command_text = transcribe_audio(audio_path)
            print(f"🎙️ [STT] Transcribed Text: {command_text}")
        except Exception as stt_error:
            return {"error": f"🎙️ STT failed: {str(stt_error)}"}

    print("\n🗣️ User said:", command_text)

    if SHOW_TIMING:
        transcriptionTime = time.time() - processTime
        print(f"⏳ [STT] Transcription Time: {round(transcriptionTime, 2)}s")
        print()
    
    print(f"🛠️ [ModuleDataExtractor] -------------------------------")

    processTime = time.time()
    # Extract module data
    module_data = ModuleDataExtractor(command_text)
    if module_data and not "error" in module_data:
        # If module data extraction is successful, proceed with the command text.
        print(f"🛠️ [ModuleDataExtractor]", module_data)
        chat_history.add_assistant(f"ModuleDataExtractor - user request context: \n"
                                    f"{json.dumps(module_data, indent=2, ensure_ascii=False)}")
    else:
        if module_data and "error" in module_data:
            print(f"🛠️ [ModuleDataExtractor] Error: {module_data['error']}")
        else:
            print(f"🛠️ [ModuleDataExtractor] No module data extracted.")
        
    if SHOW_TIMING:
        moduleExtractorTime = time.time() - processTime 
        print(f"⏳ [ModuleDataExtractor] Process Time: {round(moduleExtractorTime, 2)}s")
        print(module_data)
        print()
        processTime = time.time()


    print(f"🛠️ [ModuleDataExtractor] -------------------------------")
    print()

    # if no module data is extracted, check if chat history has module data
    if not module_data and chat_history.get_count() > 2:
        module_data = chat_history.get_module_data()
        print(f"🛠️ [ModuleDataExtractor] --->  Module Data from Chat History: {module_data}")

    # based on module select correct module_agent default to query_llm
    if module_data and module_data.get("module") == "meetings":
        # meetings_agent = MeetingsAgent(command_text, chat_history=chat_history, action=module_data.get("action"))
        # response = meetings_agent
        response = MeetingsAgent(command_text, chat_history=chat_history, action=module_data.get("action"), parameters=module_data.get("parameters", {}))
    elif module_data and module_data.get("module") == "tasks":
        print("🛠️ [TasksAgent] Not implemented yet.")
        return {"error": "Tasks module not implemented yet."}
    else:
        print("🛠️ [ModuleDataExtractor] No specific module found, using default LLM query Chat Mode.")
        # Generate the JSON command using the LLM agent, including any chat history if available.
        response = query_llm(chat_history=chat_history, command_text=command_text)
        print(f"🤖 [LLM] Generated JSON Command: {response}")

    if "error" in response:
        # Optionally, trigger a clarification/confirmation step
        response = {
            "action": "question",
            "message_to_user": "Omlouvám, nepodařilo se mi vyhodnotit tento úkol. Můžete to prosím zopakovat?"
        }

    if SHOW_TIMING:
        agentProcessTime = time.time() - processTime
        print(f"⏳ [Agent] Process Time: {round(agentProcessTime, 2)}s")
        print()
    
    
    # Validate the generated JSON command structure.
    validation = validate_json_command(response)
    if not validation["is_valid"]:
        print(f"❌ [Validation] Errors: {validation.get('errors', 'Invalid JSON command generated.')}")
        
        # if json is not valid reprompt the llm with the command text and validation error in chat history 1 attempt
        print(f"🔄 [LLM] Re-prompting LLM with command text and validation error.")
        chat_history.add_assistant(validation.get('errors', 'Invalid JSON command generated.'))
        chat_history.pretty_print()

        response = query_llm(chat_history=chat_history, command_text=validation.get('errors', 'Invalid JSON command generated.'))

        print(f"🤖 [LLM] Re-prompted JSON Command: {response}")

        # Validate the JSON command again after re-prompting.
        validation = validate_json_command(response)
        if not validation["is_valid"]:
            print(f"❌ [Validation] Errors: {validation.get('errors', 'Invalid JSON command generated.')}")
            # If the JSON command is still invalid, return an error message.
            return {
                "error": "Invalid JSON command generated.",
                "details": validation.get("errors", "Omlouvám, nepodařilo se mi vyhodnotit tento úkol. Můžete to prosím zopakovat?")
            }
        

    chat_history.pretty_print()


    # log chat history to file
    with open("logs/chat_history.log", "a", encoding="utf-8") as log_file:
        log_file.write(f"Chat History:\n{json.dumps(chat_history.get_messages(), indent=4, ensure_ascii=False)}\n\n")


    return response
