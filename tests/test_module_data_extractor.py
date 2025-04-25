# from core.llm import query_llm
from core.services.llm import query_llm
from core.agents.module_data_extractor import ModuleDataExtractor
import json

def test_module_data_extractor():
    sample_command = "Vytvořte schůzku s firmou Acmark s.r.o. zítra v 1 hodinu."

    # Call the module data extractor
    module_data = ModuleDataExtractor(sample_command)
    print("Generated JSON Command:")
    print(module_data)
    assert isinstance(module_data, dict), "The response should be a JSON object."
    assert "module" in module_data, "The JSON command should contain a 'module' field."
    assert "action" in module_data, "The JSON command should contain an 'action' field."
    # module should be meetings
    assert module_data["module"] == "meetings", "The module should be 'meetings'."
    # action should be create
    assert module_data["action"] == "create", "The action should be 'create'."
