
def validate_json_command(json_command: dict) -> dict:
    """
    Validates the structure of a JSON command.

    Args:
        json_command (dict): The JSON command to validate.

    Returns:
        dict: A dictionary indicating whether the JSON command is valid and any errors found.
    """
    required_fields = ["action", "parameters", "metadata"]
    errors = []

    # Check for required fields
    for field in required_fields:
        if field not in json_command:
            errors.append(f"Missing required field: {field}")

    # Check metadata structure
    if "metadata" in json_command:
        if not isinstance(json_command["metadata"], dict):
            errors.append("Metadata should be a dictionary.")
        else:
            if "timestamp" not in json_command["metadata"]:
                errors.append("Missing timestamp in metadata.")
            elif not isinstance(json_command["metadata"]["timestamp"], str):
                errors.append("Timestamp should be a string.")

    # Validate action and parameters
    if not isinstance(json_command.get("action"), str) or not json_command["action"]:
        errors.append("Action should be a non-empty string.")
    
    if not isinstance(json_command.get("parameters"), dict) or not json_command["parameters"]:
        errors.append("Parameters should be a non-empty dictionary.")

    return {
        "is_valid": len(errors) == 0,
        "errors": errors
    }
