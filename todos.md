# TODOs

- [x] Add more CRM actions (e.g., tasks, notes)
- [ ] Use output validation tools of JSON Schema
- [ ] Implement user and test authentication
- [ ] Integrate TTS into the main API flow
- [ ] Implement Data Auto-Ingest scheduling and monitoring
- [ ] Persistent chat history FE (fetch chat history endpoint)

### optionals
- [ ] Expand language support beyond Czech and English
- [ ] Improve error handling and logging
- [ ] Optimize embedding and search performance
- [ ] Add demo frontend interface for easier interaction
- [ ] Create Docker setup for easier deployment
- [ ] Write more comprehensive tests for edge cases
- [ ] Update documentation with more examples and usage guides
- [ ] Security review and test multi tenant isolation
- [ ] OpenAi backend option implementation




Sugar rest api
site/service/v4_1/soap.php
site/service/v4_1/rest.php

site/tests/SugarWebServices/CrudWebservicesTest.php

https://chatgpt.com/c/69899c47-4a1c-8327-a44a-3b4ab5959783

SugarCRM 6.5 api v4_1



#### TODO:  This CRM voice assistant isinteracting with SugarCRM based system, id like to expand its functionality so user can ask about client, company, contact etc.. and the llm would give him available informatin with all accesible relations etc.. how this feature should be build and integrated. Idealy getting data from CRM using SugarBean so ACLs are enforced.

https://gemini.google.com/app/03f9ea01f15cad38

To expand your CRM voice assistant's functionality to allow users to query information about clients, companies, and contacts with all accessible relations while enforcing ACLs via `SugarBean`, you should follow this multi-layered integration strategy.

### 1. Backend: Implement a "Deep Detail" CRM Endpoint

Since your current `core/adapters/crm_api.py` uses an HMAC-signed POST to a CRM gateway, you need a specific CRM-side endpoint (e.g., `/ai/v1/get-details`) that uses `SugarBean` to fetch data.

* **Logic**: Use `BeanFactory::getBean($module, $id)` to retrieve the record. Accessing fields through the bean automatically enforces SugarCRM ACLs.
* **Relations**: Explicitly load related beans (e.g., `$bean->load_relationship('contacts')`) to include associated data like primary contacts for an account or upcoming meetings for a contact.
* **Payload**: Return a structured JSON that includes the main record fields and a list of related record summaries.

### 2. Service Layer: Add a Detail Lookup Tool

In `core/services/tools.py`, implement a new tool that calls this deep detail endpoint.

```python
# core/services/tools.py

def get_record_details_tool(input_str: str) -> str:
    """
    Fetches full details and relations for a specific CRM record.
    Input: {"module": "Accounts", "id": "GUID", "tenant": "..."}
    """
    try:
        args = _parse_payload(input_str)
        # Use existing crm_api.py logic to send a 'get_details' command
        from core.adapters.crm_api import call_crm_api
        response = call_crm_api({
            "action": "get_details",
            "module": args["module"],
            "parameters": {"id": args["id"]}
        })
        return json.dumps({"success": True, "data": response})
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})

# Add to the tools list
tools.append(
    Tool(
        name="get_record_details",
        func=get_record_details_tool,
        description="Get all available info and relations for a specific record by ID."
    )
)

```

### 3. Agent Layer: Update System Prompts and Classification

You must enable the LLM to recognize "informational" requests and use the new tool.

* **Module Classification**: Update `core/agents/module_data_extractor.py` and `core/agents/schemas.py` to ensure the `get` action is correctly identified for informational queries like "Tell me about company X".
* **Agent Reasoning**: Update your module agents (e.g., `ContactsAgent` in `core/agents/modules/contacts_agent.py`) to include `get_record_details` in their available tools.
* **System Prompt Update**: Modify the system prompt to instruct the agent:
1. First, use `find_company` or `find_contact` to get the correct ID.
2. Second, use `get_record_details` with that ID to retrieve the full context including relations.
3. Finally, summarize the results for the user in the `message_to_user` field.



### 4. Integration Flow

The end-to-end data flow in `core/pipelines/command_pipeline.py` would then look like this:

1. **User Input**: "Co víš o firmě ACME?" (What do you know about ACME?)
2. **Extractor**: Identifies `module="contacts"` (or accounts) and `action="get"`.
3. **Agent**:
* Calls `find_company(query="ACME")`.
* Receives `ID: 123-abc`.
* Calls `get_record_details(module="Accounts", id="123-abc")`.
* Receives Bean data (ACL-filtered) + related contacts/tasks.


4. **Final Answer**: Formats a human-readable summary: "ACME is a client in Brno with 3 active contacts and one scheduled meeting tomorrow.".

### Key Benefits of this Approach

* **Security**: By using `SugarBean` on the CRM side, the LLM will never "see" data the user doesn't have permissions for.
* **Rich Context**: The agent can answer complex questions about relations (e.g., "Who is the primary contact there?") without multiple user turns.
* **Consistency**: It reuses your existing `run_module_agent` and `command_pipeline` architecture.