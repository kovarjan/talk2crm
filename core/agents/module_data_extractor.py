from core.services.llm import query_llm
import json

def ModuleDataExtractor(prompt: str) -> dict:
    """
    This function takes a prompt and returns the module data extracted from it.
    It uses a LLM to process the prompt and extract the module and action.
    """

    availableModules = ["meetings", "tasks", "notes", "calls"]
    availableActions = ["create", "update", "delete", "get"]
    
    # Use query_llm to call the LLM for structured output
    instruction = (
        "You are a voice assistant. Convert the user's spoken request into a JSON API call.\n"
        "Determine which module to call based on the user's request.\n"
        f"Available modules: {', '.join(availableModules)}.\n"
        "And determine the action to take.\n"
        f"Available actions: {', '.join(availableActions)}.\n"
        "Use this format strictly:\n"
        "{\n"
        "  \"module\": \"<module>\",\n"
        "  \"action\": \"<action>\"\n"
        "}"
    )

    history = [
        {
            "role": "system",
            "content": instruction
        }
    ]

    response = query_llm(prompt, history, 0.4)

    # Check if the response contains the required fields
    if 'module' not in response or 'action' not in response:
        return {"error": "Response does not contain required fields."}

    return response
