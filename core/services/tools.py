from langchain.agents import Tool
import json
from core.services.company_lookup import find_company_by_name
# from core.services.agenda import get_user_agenda  # if you have it

def find_company_tool(name: str) -> str:
    result = find_company_by_name(name)
    return json.dumps(result)

# def get_agenda_tool(input_str: str) -> str:
#     args = json.loads(input_str)
#     agenda = get_user_agenda(args["user_id"], args["date"])
#     return json.dumps(agenda)

tools = [
    Tool(
        name="find_company_by_name",
        func=find_company_tool,
        description="Search for a company in the CRM by name. Input is a string (company name)."
    ),
    # TODO: Implement this tool if you have the agenda service
    # Tool(
    #     name="get_user_agenda",
    #     func=get_agenda_tool,
    #     description="Get a user's agenda. Input is a JSON string with user_id and date."
    # )
]