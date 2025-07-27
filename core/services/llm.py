from langchain_ollama import OllamaLLM
import json
import re
from core.config import LLM_MODEL_NAME, LLM_TEMPERATURE
from core.utils.chat import ChatSession

# Initialize the Ollama model once (reuse across calls ideally)
llm = OllamaLLM(model=LLM_MODEL_NAME, temperature=LLM_TEMPERATURE)

def query_llm(
    chat_history: ChatSession,
    command_text: str = None,
    temperature=LLM_TEMPERATURE,
    add_history: bool = True,
    returnJson: bool = True,
) -> dict:
    # Ensure a ChatSession instance exists
    if not chat_history:
        chat_history = ChatSession()
        if command_text:
            chat_history.add_user(command_text)

    if chat_history and command_text:
        chat_history.add_user(command_text)

    # Convert to LangChain-compatible messages
    langchain_messages = chat_history.to_langchain_messages()

    try:
        # Call Ollama via LangChain
        output_text = llm.invoke(langchain_messages)

        # Log response
        with open("logs/llm_response.log", "a", encoding="utf-8") as log_file:
            log_file.write(f"LLM Response:\n{output_text}\n\n")

        # Clean up <think> tags
        output_text = re.sub(r"<think>.*?</think>", "", output_text, flags=re.DOTALL)
        output_text = re.sub(r"<think\s*/?>", "", output_text)

        if returnJson:
            # Strip code fences, comments, and fix JSON format
            output_text = re.sub(r"^```json|^```|```$", "", output_text.strip(), flags=re.MULTILINE)
            output_text = re.sub(r"//.*?$|#.*?$", "", output_text, flags=re.MULTILINE)
            output_text = re.sub(r",\s*}", "}", output_text)
            output_text = re.sub(r",\s*]", "]", output_text)

            result = json.loads(output_text)
        else:
            result = output_text

        # Add response to chat history
        if add_history:
            if returnJson:
                chat_history.add_llm_response(result)
            else:
                chat_history.add_assistant(output_text)

    except Exception as e:
        with open("logs/llm_response.log", "a", encoding="utf-8") as log_file:
            log_file.write(f"LLM Error: Failed to generate JSON command\n{str(e)}\n")
        result = {"error": f"Failed to generate JSON command: {str(e)}"}

    return result
