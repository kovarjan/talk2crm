from jsonschema import validate, ValidationError


def validate_json_command(json_command: dict, schema: dict = None) -> dict:
    """
    Validates the structure of a JSON command.

    Args:
        json_command (dict): The JSON command to validate.

    Returns:
        dict: A dictionary indicating whether the JSON command is valid and any errors found.
    """
    errors = []

    # {
    #     "action": "create",
    #     "module": "calls",
    #     "parameters": {
    #         "subject": "Projekt Nová Kampaň",
    #         "contact_name": "Jana Malinová"
    #     },
    #     "metadata": {
    #         "timestamp": "2025-04-26T08:40:20Z"
    #     }
    # }

    if schema is None:
        # Define a basic schema for validation
        schema = {
            "type": "object",
            "properties": {
                "action": {"type": "string"},
                "message_to_user": {"type": "string"},
                "parameters": {"type": "object"},
                "metadata": {"type": "object"},
            },
            "required": ["action"],
            "if": {"properties": {"action": {"const": "delete"}}},
            "then": {},
            "else": {"required": ["message_to_user"]},
        }

    # Check if the JSON command is empty
    if not json_command:
        errors.append("JSON command is empty.")
        return {"is_valid": False, "errors": errors}
    # Check if the JSON command is a dictionary
    if not isinstance(json_command, dict):
        errors.append("JSON command should be a dictionary.")
        return {"is_valid": False, "errors": errors}

    # Validate the JSON command structure by schema
    try:
        # Validate the JSON command against the schema
        validate(instance=json_command, schema=schema)
    except ValidationError as e:
        errors.append(f"Validation error: {e.message}")
        # If schema validation fails, add the error message to the errors list
        return {"is_valid": False, "errors": errors}

    return {"is_valid": len(errors) == 0, "errors": errors}
