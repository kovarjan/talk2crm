import json
import re
from core.services.llm import query_llm
from core.config import LLM_TEMPERATURE, LLM_MODEL_NAME
from core.utils.json_validator import validate_json_command
from core.utils.chat import ChatSession
from core.services.tools import tools
from langchain.agents import initialize_agent, AgentType
from langchain_ollama import ChatOllama   # ← chat model is better for ReAct
from core.services.spellcheck import correct_text

FUZZY_MATCH_THRESHOLD = 0.6

# --- Helpers -----------------------------------------------------------------

# --- robust parsing helpers ---

import json, re

FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE | re.MULTILINE)
THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)

def _looks_like_json(s: str) -> bool:
    s = s.strip()
    return s.startswith("{") and s.endswith("}")

def _extract_balanced_json(text: str) -> dict:
    """Scan for first balanced {...} ignoring braces inside strings."""
    text = THINK_RE.sub("", text)
    text = FENCE_RE.sub("", text).strip()

    start = text.find("{")
    if start == -1:
        raise ValueError("No '{' found in output.")
    i = start
    depth = 0
    in_string = False
    escape = False
    buf = []

    while i < len(text):
        ch = text[i]
        buf.append(ch)

        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
        else:
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    break
        i += 1

    s = "".join(buf).replace(",}", "}").replace(",]", "]")
    return json.loads(s)

def parse_agent_output(output_obj) -> dict:
    """
    Accepts whatever LangChain returned (dict or str) and returns a dict.
    Priority:
      1) If dict['output'] is pure JSON -> json.loads
      2) If string has 'Final Answer:' -> parse JSON after it
      3) Else -> extract first balanced JSON block
    """
    # 1) Dict from AgentExecutor
    if isinstance(output_obj, dict) and "output" in output_obj:
        s = str(output_obj["output"]).strip()
        if _looks_like_json(s):
            return json.loads(s)
        # else fall through to string handling with the same s
        output_str = s
    else:
        output_str = str(output_obj)

    # 2) Try 'Final Answer:' path
    body = THINK_RE.sub("", output_str)
    if "Final Answer:" in body:
        ans = body.split("Final Answer:", 1)[1]
        ans = FENCE_RE.sub("", ans).strip()
        ans = ans.replace(",}", "}").replace(",]", "]")
        try:
            return json.loads(ans)
        except Exception:
            pass  # fall back to balanced extraction

    # 3) First balanced JSON
    return _extract_balanced_json(output_str)

# --- LLM & Agent -------------------------------------------------------------

llm = ChatOllama(
    model=LLM_MODEL_NAME,       # prefer an *instruct* tag of Qwen if available
    temperature=LLM_TEMPERATURE # keep low for determinism
    # NOTE: do NOT set format="json" for ReAct agents; they need to speak Thought/Action/Observation
)

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,   # ReAct agent
    verbose=True,
    handle_parsing_errors=True,                    # let it self-correct once
    max_iterations=6,                              # a bit more room to finish tool calls
)

# --- Meetings Agent ----------------------------------------------------------

def MeetingsAgent(
    command_text: str,
    chat_history: ChatSession,
    action: str = "",
    module_context: dict = None,
) -> dict:
    print(f"🤖 [MeetingsAgent] Command Text: {command_text}")

    if not chat_history:
        chat_history = ChatSession()

    chat_history.add_user(command_text)

    # Build optional context block
    context_str = ""
    if module_context:
        context_items = [f"{k}: {v}" for k, v in module_context.items()]
        context_str = "CONTEXT:\n" + "\n".join(context_items) + "\n"

    # ReAct-compliant instructions: allow Thought/Action… but REQUIRE Final Answer JSON.
    schema_prompt = f"""
{context_str}
You are a CRM meetings agent. Use tools (e.g., find_company_by_name, get_user_agenda) as needed.
Default meeting duration is 60 minutes. Check user's agenda for conflicts.

Follow this exact interaction format while reasoning:
Thought: describe what you will do
Action: <tool name>   (only when you need to call a tool)
Action Input: <string or JSON for the tool>
Observation: <tool result>

(Repeat Thought/Action/Action Input/Observation as needed.)

When you have all necessary information, produce the final output on a new line:

Final Answer: {{"action":"{action}","module":"meetings","parameters":{{"name":string|null,"related_to":string|null,"related_to_id":string|null,"related_module":"contacts|companies|users"|null}},"metadata":{{"date":"YYYY-MM-DD"|null,"time":"HH:MM"|null,"duration":number|null,"participants":[string]|null,"location":string|null}},"message_to_user":string|null}}

Rules:
- The ONLY content after 'Final Answer:' must be ONE valid JSON object matching the schema.
- Do NOT include analysis or any extra text after 'Final Answer:'.
- If required data is missing, set action to "question" and return a brief Czech question in "message_to_user".
"""
    chat_history.add_system(schema_prompt)

    # Run agent
    try:
        langchain_messages = chat_history.to_langchain_messages()
        output = agent.invoke(langchain_messages)
    except Exception as e:
        print(f"❌ Agent failed: {e}")
        return {
            "action": "error",
            "message_to_user": "Omlouvám se, došlo k chybě při zpracování požadavku. Zkuste to prosím znovu."
        }

    # Parse the output as JSON (robustly)
    try:
        # output_text = output  # Assign output to output_text for logging and parsing
        print(f"🤖 [MeetingsAgent] Raw Output: {output}")
        result = parse_agent_output(output)
    except Exception as e:
        print(f"❌ JSON parse error: {e}")
        return {
            "action": "question",
            "message_to_user": "Omlouvám se, nerozumím přesně zadání. Můžete upřesnit, koho a kdy mám naplánovat?"
        }

    # Validate against your schema
    validation = validate_json_command(result)
    if not validation["is_valid"]:
        print(f"❌ Validation error: {validation['errors']}")
        return {
            "action": "error",
            "message_to_user": "Omlouvám se, požadavek není úplný. Můžete ho prosím upřesnit?"
        }

    # Spellcheck message_to_user
    if "message_to_user" in result and result["message_to_user"]:
        print(f"🤖 [MeetingsAgent] Result message: {result['message_to_user']}")
        result["message_to_user"] = correct_text(result["message_to_user"])
        print(f"🤖 [MeetingsAgent] Corrected message: {result['message_to_user']}")

    chat_history.add_llm_response(result)
    return result
