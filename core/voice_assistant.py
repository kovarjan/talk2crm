import whisper
# from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from core.services.llm import query_llm
import datetime
import json
import os
import re
from pydub import AudioSegment

# Load whisper model
whisper_model = whisper.load_model("base")

# Convert MP3 to WAV
def convert_mp3_to_wav(mp3_path: str, wav_path: str):
    audio = AudioSegment.from_mp3(mp3_path)
    audio.export(wav_path, format="wav")

# Transcribe voice command
def transcribe_voice(mp3_path: str) -> str:
    wav_path = "temp.wav"
    convert_mp3_to_wav(mp3_path, wav_path)
    result = whisper_model.transcribe(wav_path)
    return result['text']

def clean_json_string(text: str) -> str:
    # Remove // comments
    text = re.sub(r"//.*", "", text)
    # Remove /* */ comments
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    return text.strip()

# NOTE: Tyto moduly by se neměly volat na kompletním chatu četně response, mate to model.
# use llm to determine wich module to call
def get_module(prompt: str|list) -> str:
    instruction = (
        "You are a voice assistant. Convert the user's spoken request into a JSON API call.\n"
        "Determine which module to call based on the user's request.\n"
        "Available modules: meetings, tasks, notes, calls.\n"
        "And determine the action to take.\n"
        "Available actions: create, update, delete.\n"
        "Use this format strictly:\n"
        "{\n"
        "  \"module\": \"<module>\",\n"
        "  \"action\": \"<action>\"\n"
        "}"
    )


    response = query_llm(prompt, instruction, 0.4)
    print("\n🛠️ > ModuleData full:", response)

    # strip <think> data from response
    response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL)
    # strip <think> data from response
    response = re.sub(r"<think/>", "", response, flags=re.DOTALL)
    # strip <think> data from response
    response = re.sub(r"<think>", "", response, flags=re.DOTALL)
    # strip <think> data from response

    return response

# Use llm.py to call the LLM for structured output
def get_api_call(promptMessages: list) -> str:


    filteredMessages = []
    for message in promptMessages:
        if message['role'] == 'user':
            filteredMessages.append(message)

    print("\n\n--> > promptMessages:" )
    for message in promptMessages:
        print(message)
        # print("   > " + message['role'] + ": " + message['message'])
    print("\n\n--> > filteredMessages:")
    for message in filteredMessages:
        print(message)
        # print("   > " + message['role'] + ": " + message['message'])

    moduleData = get_module(filteredMessages)
    print("\n🛠️ > ModuleData:", moduleData)
    try:
        moduleData = json.loads(clean_json_string(moduleData))
    except json.JSONDecodeError:
        return "⚠️ Sorry, the local LLM failed to respond."
    if 'module' not in moduleData or 'action' not in moduleData:
        return "⚠️ Sorry, the local LLM failed to respond."
    
    module = moduleData['module']
    action = moduleData['action']

    print("   > Module:", module)
    print("   > Action:", action)

    # Check if the module is one of the supported ones
    if module not in ["meetings", "tasks", "notes", "calls"]:
        return "⚠️ Sorry, the local LLM failed to respond."    
    
    if module == "meetings":
        instruction = (
            "You are a voice assistant. Convert the user's spoken request into a JSON API call.\n"
            "If any key information is missing, respond with a question to clarify.\n" # add day of the week
            "Dates can be derived from the current date. Now: " + datetime.datetime.now().strftime("%Y-%m-%d") + " day: " + datetime.datetime.now().strftime("%A") + "\n"
            "Response format is always json with params: 'response: <response confirmation or question> and api call params'\n"
            "No json comments are allowed.'\n"
            "Required fields are: 'action', 'module', 'name', 'date', 'with'.\n"
            "For confirmation, use 'mode: confirm' and response 'Potvrďte prosím následující údaje.'.\n"
            "Use this format strictly:\n"
            "{\n"
            "  \"response\": \"<response confirmation or question to user if you dont have all information>\",\n"
            "  \"api_call\": {\n"
            "     \"action\": \"" + action + "\",\n"
            "     \"module\": \"" + module + "\",\n"
            "     \"name\": \"Schůzka s <company/user/contact>\",\n"
            "     \"with\": \"<company/user/contact>\",\n"
            "     \"date\": \"<date in format YYYY-MM-DD>\",\n"
            "     \"time\": \"<time if mentioned>\",\n"
            "     \"duration\": \"<duration if mentioned>\",\n"
            "     \"location\": \"<location if mentioned>\"\n"
            "  },\n"
            "  \"mode\": \"<'confirm' information or 'clarify' more>\"\n"
            "}"
        )

        # full_prompt = f"{instruction}\n\nUser said: {prompt}"
        full_prompt = [
            {"role": "system", "content": instruction},
            *promptMessages,
        ]
        return query_llm(full_prompt, instruction)

    return "⚠️ Sorry, unsupported module. Not implemented yet."
