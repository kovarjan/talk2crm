from langchain.agents import Tool
from core.services.spellcheck import correct_text
import json
from core.services.company_lookup import find_company_by_name
from core.services.contacts_lookup import find_contact_by_name_or_email
# from core.services.agenda import get_user_agenda  # if you have it

def find_company_tool(name: str) -> str:
    result = find_company_by_name(name)
    return json.dumps(result)

def find_contact_tool(query: str) -> str:
    result = find_contact_by_name_or_email(query)
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
    Tool(
        name="find_contact_by_name_or_email",
        func=find_contact_tool,
        description="Search for a contact in the CRM by name or email. Input is a string (name or email)."
    ),
    # TODO: Implement this tool if you have the agenda service
    # Tool(
    #     name="get_user_agenda",
    #     func=get_agenda_tool,
    #     description="Get a user's agenda. Input is a JSON string with user_id and date."
    # )
    # Tool(
    #     name="LocalCzechSpellchecker",
    #     func=correct_text,
    #     description="Locally fixes Czech spelling using Hunspell/Spylls."
    # ),
]
