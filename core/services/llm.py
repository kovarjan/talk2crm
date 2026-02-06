# Centralizes all LLM-related logic: chat JSON calls, robust JSON parsing,
# and ReAct agent execution (used by MeetingsAgent).

from __future__ import annotations

import json
import re
from functools import lru_cache
from typing import Any, Dict, Optional

from langchain_ollama import ChatOllama
from langchain_core.tools import Tool


from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.agents import AgentExecutor
from langchain_classic.agents.react.agent import create_react_agent

from core.config import LLM_MODEL_NAME, LLM_TEMPERATURE, DEBUG_LLM, DISABLE_REASONING
from core.utils.chat import ChatSession
from core.utils.json_validator import validate_json_command
from core.services.spellcheck import correct_text
from core.services.tools import tools

# ------------------------ Robust JSON parsing helpers -------------------------

FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE | re.MULTILINE)
THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)

def _strip_meta(s: str) -> str:
    """Remove <think> tags, code fences; trim and normalize trailing commas."""
    s = THINK_RE.sub("", s or "")
    s = FENCE_RE.sub("", s).strip()
    # normalize some sloppy JSON tails the model sometimes emits
    s = s.replace(",}", "}").replace(",]", "]")
    return s

def _looks_like_json(s: str) -> bool:
    s = (s or "").strip()
    return s.startswith("{") and s.endswith("}")

def _extract_balanced_json(text: str) -> Dict[str, Any]:
    """
    Extract the first balanced {...} from the given text (robust to quotes/escapes).
    """
    text = _strip_meta(text)
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
                    i += 1  # include the closing brace
                    break
        i += 1

    # ---- post-loop (only now we should parse/return) ----
    if depth != 0:
        raise ValueError("Unbalanced braces in output.")

    s = "".join(buf).replace(",}", "}").replace(",]", "]")
    obj = json.loads(s)

    # optional: hoist message_to_user if nested
    for key in ("parameters", "metadata"):
        if isinstance(obj.get(key), dict) and "message_to_user" in obj[key]:
            obj["message_to_user"] = obj[key].pop("message_to_user")

    return obj

def _extract_final_json(text: str, marker: str = "Final Answer:") -> Optional[dict]:
    body = THINK_RE.sub("", text or "")
    pos = body.rfind(marker)
    if pos == -1:
        return None

    tail = _strip_meta(body[pos + len(marker):])

    # strip anything before the first '{' (e.g., stray words/newlines)
    brace = tail.find("{")
    if brace != -1:
        tail = tail[brace:]

    if _looks_like_json(tail):
        try:
            return json.loads(tail)
        except Exception:
            pass

    try:
        return _extract_balanced_json(tail)
    except Exception:
        return None

def _normalize_meeting_command(cmd: dict) -> dict:
    """
    Light schema cleanup so the gateway gets consistent payloads.
    - ensure objects exist
    - map 'company'/'companies' -> 'accounts'
    - dedupe participant_ids and participants
    - trim time HH:MM:SS -> HH:MM
    """
    if not isinstance(cmd, dict):
        return cmd

    action  = cmd.get("action")
    module  = cmd.get("module")

    # Only normalize meetings create
    # Only normalize for certain modules/actions
    allowed_modules = {"meetings", "contacts", "calls"}
    if action != "create" or module not in allowed_modules:
        return cmd

    params = cmd.setdefault("parameters", {})
    meta   = cmd.setdefault("metadata", {})

    # related_module normalization
    rel_mod = params.get("related_module")
    if rel_mod:
        m = str(rel_mod).lower().strip()
        if m in ("company", "companies"):
            params["related_module"] = "accounts"
        elif m in ("contact", "contacts"):
            params["related_module"] = "contacts"
        elif m in ("user", "users"):
            params["related_module"] = "users"
    else:
        # if we have an id but no module, assume accounts (company)
        if params.get("related_to_id"):
            params["related_module"] = "accounts"

    # participant_ids / participants dedupe
    if isinstance(meta.get("participant_ids"), list):
        meta["participant_ids"] = sorted({pid for pid in meta["participant_ids"] if pid})
    if isinstance(meta.get("participants"), list):
        meta["participants"] = sorted({p for p in meta["participants"] if p})

    # time: trim seconds if present
    t = meta.get("time")
    if isinstance(t, str) and len(t) >= 5:
        parts = t.split(":")
        if len(parts) >= 2:
            meta["time"] = f"{parts[0]:0>2}:{parts[1]:0>2}"

    # date: very light format check
    d = meta.get("date")
    if isinstance(d, str):
        d = d.strip()
        if d and d.count("-") == 2:
            meta["date"] = d

    return cmd

def parse_agent_output(output_obj: Any) -> Dict[str, Any]:
    """
    Accepts whatever LangChain AgentExecutor returned (dict or str) and returns a dict.
    Priority:
      1) If dict['output'] present -> parse 'Final Answer:' tail first, then raw JSON, then balanced JSON
      2) For strings -> same order
    Finally, apply light schema normalization for meetings.
    """
    # Dict (typical AgentExecutor result)
    if isinstance(output_obj, dict) and "output" in output_obj:
        s = str(output_obj["output"]).strip()

        # Prefer tolerant Final Answer parser
        j = _extract_final_json(s)
        if j is not None:
            return _normalize_meeting_command(j)

        # If its pure JSON, accept it
        if _looks_like_json(s):
            try:
                return _normalize_meeting_command(json.loads(s))
            except Exception:
                pass

        # Fallback: first balanced JSON anywhere
        try:
            j = _extract_balanced_json(s)
            return _normalize_meeting_command(j)
        except Exception:
            return {"action": "error", "message_to_user": "Omlouvám se, výstup se nepodařilo zpracovat. Zkuste to prosím znovu. (A1)"}

    # String response
    s = _strip_meta(str(output_obj))

    j = _extract_final_json(s)
    if j is not None:
        return _normalize_meeting_command(j)

    if _looks_like_json(s):
        try:
            return _normalize_meeting_command(json.loads(s))
        except Exception:
            pass

    try:
        j = _extract_balanced_json(s)
        return _normalize_meeting_command(j)
    except Exception:
        return {"action": "error", "message_to_user": "Omlouvám se, výstup se nepodařilo zpracovat. Zkuste to prosím znovu. (A2)"}


# -------------------------- Model singletons (cached) -------------------------

@lru_cache(maxsize=1)
def get_chat_llm(temperature: float = LLM_TEMPERATURE) -> ChatOllama:
    # Single chat model used for both simple JSON and ReAct
    return ChatOllama(model=LLM_MODEL_NAME, temperature=temperature)


# ------------------------------ Simple JSON call ------------------------------

def query_llm(
    chat_history: Optional[ChatSession],
    command_text: Optional[str] = None,
    temperature: float = LLM_TEMPERATURE,
    add_history: bool = True,
    returnJson: bool = True,
    is_assistant_prompt: bool = False,
    verbose: bool = DEBUG_LLM,
    no_thinking: bool = True,
) -> Dict[str, Any]:
    """
    Unified simple call: one chat completion. By default, expects JSON and parses robustly.
    - Uses the shared ChatOllama chat model
    - No tools are invoked here (tool-free)
    - Robust JSON parsing via _extract_balanced_json

    Returns a dict (or {"error": "..."} on failure).
    """
    try:
        if not chat_history:
            chat_history = ChatSession()

        # Strengthen the system instruction for extractor-style calls.
        # This discourages chain-of-thought and enforces JSON-only.
        if no_thinking:
            chat_history.add_system(
                "You are a strict JSON generator. "
                "Do NOT include chain-of-thought, analysis, or explanations. "
                "Output ONLY a single JSON object."
                " /no_think"
            )

        if command_text:
            if is_assistant_prompt:
                chat_history.add_assistant(command_text)
            else:
                chat_history.add_user(command_text)

        # Convert to LangChain messages and run
        messages = chat_history.to_langchain_messages()
        print(f"🤖 Querying LLM with {len(messages)} messages...")
        for msg in messages:
            print(msg)
        resp = get_chat_llm(temperature=temperature).invoke(messages)

        # Normalize to string
        output_text = getattr(resp, "content", None) or str(resp)
        output_text = _strip_meta(output_text)

        if not returnJson:
            if add_history:
                chat_history.add_assistant(output_text)
            return {"text": output_text}

        # --- tolerant JSON parsing (A) ---
        # If theres a 'Final Answer:' tail, use that
        j = _extract_final_json(output_text)
        if j is not None:
            if add_history:
                chat_history.add_llm_response(j)
            return j

        # If output is pure JSON, accept it
        try:
            if _looks_like_json(output_text):
                result = json.loads(output_text)
            else:
                # Fallback: first balanced JSON anywhere in the text
                result = _extract_balanced_json(output_text)
        except Exception:
            return {"error": "Model did not return JSON."}

        if add_history:
            chat_history.add_llm_response(result)

        return result


    except Exception as e:
        return {"error": f"Failed to generate JSON command: {str(e)}"}


# ------------------------------- ReAct Agent API ------------------------------

@lru_cache(maxsize=16)
def _react_agent(tenant: str):
    # Generic ReAct wrapper; module-specific schema/instructions come via chat_history
    # tenant-bound tools list
    ttools = _tenant_tools(tenant)

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a helpful CRM assistant. Use tools when helpful.\n\n"
            "Available tools:\n{tools}\n\n"
            "Tool names you can call: {tool_names}\n"
            "Follow STRICTLY this loop when using tools:\n"
            "Thought:\n"
            "Action: <one of {tool_names}>\n"
            "Action Input: <tool input as plain string or JSON>\n"
            "Observation: <tool output>\n\n"
            "When you have everything, output exactly one line starting with:\n"
            "Final Answer: must be ONE JSON object only nothing else\n"
            "Do not include anything after the JSON."
            "Absolutely no text or commentary between the last Observation and \"Final Answer:\". If you include any other text, the run fails.\n"
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
        ("ai", "{agent_scratchpad}"),
    ])

    agent = create_react_agent(llm=get_chat_llm(), tools=ttools, prompt=prompt)

    executor = AgentExecutor(
        agent=agent,
        tools=ttools,
        verbose=DEBUG_LLM,
        max_iterations=10,
        handle_parsing_errors=(
            "Your previous message was not in the required ReAct format.\n"
            "Either continue with:\n"
            "  Action:\n"
            "  Action Input:\n"
            "…or finish with:\n"
            "Final Answer: must be ONE JSON object only nothing else\n"
            "Do not include anything after the JSON."
            "Absolutely no text or commentary between the last Observation and \"Final Answer:\". If you include any other text, the run fails.\n"
        ),
        # early_stopping_method removed due to incompatibility
        return_intermediate_steps=False,
    )

    return executor

def run_react_agent(chat_history: ChatSession, tenant: str) -> Dict[str, Any]:
    try:
        messages = chat_history.to_langchain_messages()
        print(f">> Running ReAct agent with {len(messages)} messages... [tenant={tenant}]")

        last_user = next((m.content for m in reversed(messages) if m.type == "human"), "")
        output = _react_agent(tenant).invoke({
            "chat_history": messages,
            "input": last_user,
        })
    except Exception as e:
        print(f"❗ Agent execution error: {e}")
        return {
            "action": "error",
            "message_to_user": "Omlouvám se, došlo k chybě při zpracování požadavku. Zkuste to prosím znovu."
        }

    try:
        return parse_agent_output(output)
    except Exception:
        print("❗ Failed to parse agent output.")
        print(f"Raw output: {output}")
        return {
            "action": "question",
            "message_to_user": "Omlouvám se, nerozumím přesně zadání. Můžete upřesnit, koho a kdy mám naplánovat?"
        }

# -------------------------- Meetings-specific wrapper -------------------------

def run_module_agent(
    command_text: str,
    chat_history: Optional[ChatSession],
    system_prompt: str = None,
    module_context: Optional[dict] = None,
    tenant: str = "unknown",
) -> Dict[str, Any]:
    """
    A thin wrapper that prepares the schema/system prompt for meetings and
    then executes the shared ReAct agent. Validates and post-processes
    the final JSON, including Czech spellcheck for message_to_user.
    """
    if not chat_history:
        chat_history = ChatSession()

    chat_history.add_user(command_text)

    # Context block (optional)
    if module_context is None:
        module_context = {}
    module_context.setdefault("tenant", tenant)

    context_str = ""
    if module_context:
        context_items = [f"{k}: {v}" for k, v in module_context.items()]
        context_str = "CONTEXT:\n" + "\n".join(context_items) + "\n"

    if system_prompt:
        system_prompt = f"""{context_str}
        {system_prompt}"""

    chat_history.add_system(system_prompt)

    # Execute shared agent with tenant
    result = run_react_agent(chat_history, tenant=tenant)

    # Validate against global schema
    validation = validate_json_command(result)
    if not validation.get("is_valid"):
        return {
            "action": "error",
            "message_to_user": "Omlouvám se, požadavek není úplný. Můžete ho prosím upřesnit?"
        }

    # Spellcheck message_to_user if present
    mtu = result.get("message_to_user")
    if mtu:
        result["message_to_user"] = correct_text(mtu)

    chat_history.add_llm_response(result)
    return result

# -------------------------- Tenant-aware tools -------------------------------

def _wrap_tool_with_tenant(t: Tool, tenant: str) -> Tool:
    """Return a new Tool that injects tenant into JSON inputs (if absent/unknown)."""
    def _call(s: str):
        # Most of your tools expect JSON string input; be liberal but safe:
        try:
            data = json.loads(s) if isinstance(s, str) and s.strip().startswith(("{", "[")) else {}
        except Exception:
            data = {}

        # Inject tenant if missing/unknown
        if not isinstance(data, dict):
            data = {"input": data}
        if str(data.get("tenant", "")).strip() in ("", "unknown", "None", "null"):
            data["tenant"] = tenant

        # Re-serialize back to the original tool contract (string input)
        payload = json.dumps(data, ensure_ascii=False)
        return t.func(payload)

    return Tool(
        name=t.name,
        description=t.description,
        func=_call,
    )


def _tenant_tools(tenant: str):
    """Clone your global tools list, binding each func to provided tenant."""
    return [ _wrap_tool_with_tenant(t, tenant) for t in tools ]
