
from core.pipelines.command_pipeline import run_command_pipeline

def test_run_command_pipeline():
    # Sample audio path (replace with actual audio file for real testing)
    audio_path = "assets/audio/untitled_exmaple2.mp3"
    
    # Process the voice command
    response = run_command_pipeline(audio_path)
    
    # Print the response for debugging
    print("Response from run_command_pipeline:")
    print(response)
    
    # Validate the response structure
    assert isinstance(response, dict), "The response should be a JSON object."
    assert "module" in response, "The JSON command should contain a 'module' field."
    assert "action" in response, "The JSON command should contain an 'action' field."
    assert "parameters" in response, "The JSON command should contain a 'parameters' field."
    assert "metadata" in response, "The JSON command should contain a 'metadata' field."
    assert "timestamp" in response["metadata"], "The metadata should contain a 'timestamp' field."
    assert isinstance(response["metadata"]["timestamp"], str), "The timestamp should be a string."
    assert response["action"] != "", "The action field should not be empty."
    assert isinstance(response["parameters"], dict), "The parameters field should be a dictionary."
    assert len(response["parameters"]) > 0, "The parameters field should not be empty."
    assert isinstance(response["metadata"], dict), "The metadata field should be a dictionary."
    assert len(response["metadata"]) > 0, "The metadata field should not be empty."
    assert response["metadata"]["timestamp"] != "", "The timestamp field should not be empty."
