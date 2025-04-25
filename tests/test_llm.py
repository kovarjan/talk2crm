# from core.llm import query_llm
from core.services.llm import query_llm
import json

def test_query_llm():
    sample_command = "Vytvořte schůzku s Janem Novákem na zítra."
    
    # Example chat history with a previous clarification or conversational context.
    history = [
        {
            "role": "assistant",
            "content": "V kolik hodin byste chtěl mít schůzku?"
        },
        {
            "role": "user",
            "content": "V 10 hodin."
        }
    ]
    
    json_command = query_llm(sample_command, chat_history=history)
    print("Generated JSON Command:")
    print(json.dumps(json_command, indent=4))

    assert isinstance(json_command, dict), "The response should be a JSON object."
    assert "action" in json_command, "The JSON command should contain an 'action' field."
    assert "parameters" in json_command, "The JSON command should contain a 'parameters' field."
    assert "metadata" in json_command, "The JSON command should contain a 'metadata' field."
    assert "timestamp" in json_command["metadata"], "The metadata should contain a 'timestamp' field."
    assert isinstance(json_command["metadata"]["timestamp"], str), "The timestamp should be a string."
    assert json_command["action"] != "", "The action field should not be empty."
    assert isinstance(json_command["parameters"], dict), "The parameters field should be a dictionary."
    assert len(json_command["parameters"]) > 0, "The parameters field should not be empty."
    assert isinstance(json_command["metadata"], dict), "The metadata field should be a dictionary."
    assert len(json_command["metadata"]) > 0, "The metadata field should not be empty."
    assert json_command["metadata"]["timestamp"] != "", "The timestamp field should not be empty."
