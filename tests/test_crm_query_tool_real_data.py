import os
import json
import pytest
import asyncio

from app.services.crm_client import CoripoClient
from app.engine.tools import build_tools

@pytest.mark.asyncio
async def test_crm_query_tool_real_data():
    
    real_client = CoripoClient(
        base_url=os.getenv("CORIPO_TEST_BASE_URL"),
        token=os.getenv("CORIPO_TEST_TOKEN"),
        user_id=os.getenv("CORIPO_TEST_USER_ID"),
        user_name=os.getenv("CORIPO_TEST_USER_NAME"),
    )

    print("\nTesting CRM Query Tool with real data...")

    # Instantiate the tools with real dependencies
    tools = build_tools(
        tenant_id="local_dev_tenant",
        user_id="local_dev_user",
        input_text="",
        request_context=None,
        crm_client=real_client,
        rag_service=None,
        action_confirmation=False,
    )

    # Extract the query tool (assuming it's the third one in the list)
    query_tool = tools[3]

    payload = {
        "module": "Contacts",
        "filters": json.dumps([
            {
                "field": "account_id",
                "op": "eq",
                "value": "a42333d4-c035-2f73-865c-64ee30906163"
            }
        ]),
        "limit": 50
    }

    result_str = await query_tool.arun(payload)
    parsed_result = json.loads(result_str)

    print("Asserting the structure of the result...\n\n\n")
    print(json.dumps(parsed_result, indent=2))


    assert isinstance(parsed_result, (list, dict))

    # should include PANAS, spol. s r.o.
    # should include Jan Mimra
    # should include jmimra@panas.cz
    # should include mimrova@panas.cz
    # should include pekarkova@panas.cz

    assert "Jan" in parsed_result['summary']
    assert "Mimra" in parsed_result['summary']
    assert "PANAS, spol. s r.o." in parsed_result['summary']

    assert any(card["meta"].get("E-mail") == "jmimra@panas.cz" for card in parsed_result["cards"])
    assert any(card["meta"].get("E-mail") == "mimrova@panas.cz" for card in parsed_result["cards"])
    assert any(card["meta"].get("E-mail") == "pekarkova@panas.cz" for card in parsed_result["cards"])
