from core.services.llm import query_llm
from core.utils.chat import ChatSession


def ModuleDataExtractor(prompt: str, chat_history: ChatSession = None) -> dict:
    """
    This function takes a prompt and returns the module data extracted from it.
    It uses a LLM to process the prompt and extract the module and action.
    """

    availableModules = ["meetings", "tasks", "notes", "calls"]
    availableActions = ["create", "update", "delete", "get"]
    availableSubjects = [
        "company",
        "contact",
        "user",
        "task",
        "note",
        "call",
        "meeting",
        "invoice",
    ]

    # Use query_llm to call the LLM for structured output
    instruction = (
        "Determine which module to call based on the user's request.\n"
        f"Available modules: {', '.join(availableModules)}.\n"
        "And determine the action to take.\n"
        f"Available actions: {', '.join(availableActions)}.\n"
        "The subject of the request should be name of a record in following related modules: "
        f"{', '.join(availableSubjects)}.\n"
        "Use this JSON format strictly:\n"
        "{\n"
        '  "module": "<module>",\n'
        '  "action": "<action>"\n'
        '  "parameters": {\n'
        '    "related_module": "<related_module>",\n'
        '    "related_name": "<related_module>",\n'
        "  }\n"
        "} \n"
        "If the request is not clear, return empty structure.\n"
        "Respond ONLY with specified JSON. Do NOT return property message_to_user.\n"
    )
    
    with open("logs/llm_response.log", "a", encoding="utf-8") as log_file:
        log_file.write(f"ModuleDataExtractor\n")

    history = None

    if not chat_history:
        # Initialize a new chat session if no history is provided
        history = ChatSession(system_prompt=instruction)
    else:
        history = ChatSession(system_prompt=instruction)
        history.load_history(chat_history.get_user_assistant_messages())

    response = query_llm(history, prompt, 0.4)

    history.pretty_print(False, 8)

    # Check if the response contains the required fields
    if "module" not in response or "action" not in response:
        print("❗ [ModuleDataExtractor] Response does not contain required fields.")
        print("❗ Response:", response)
        return None
    
    return response
