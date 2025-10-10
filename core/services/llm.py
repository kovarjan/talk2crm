# core/services/llm.py  (UNIFIED)
# Centralizes all LLM-related logic: chat JSON calls, robust JSON parsing,
# and ReAct agent execution (used by MeetingsAgent).
#
# Drop-in replacement for the previous llm.py and meetings agent LLM parts.

from __future__ import annotations

import json
import re
from functools import lru_cache
from typing import Any, Dict, Optional

from langchain_ollama import ChatOllama
# from langchain.agents import initialize_agent, AgentType
from langchain.agents import create_react_agent, AgentExecutor
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

from core.config import LLM_MODEL_NAME, LLM_TEMPERATURE, DEBUG_LLM, DISABLE_REASONING
from core.utils.chat import ChatSession
from core.utils.json_validator import validate_json_command
from core.services.spellcheck import correct_text
from core.services.tools import tools
import pprint

# ------------------------ Robust JSON parsing helpers -------------------------

FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE | re.MULTILINE)
THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)

def _strip_meta(s: str) -> str:
    """Remove <think> tags, code fences; trim and normalize."""
    s = THINK_RE.sub("", s)
    s = FENCE_RE.sub("", s).strip()
    s = s.replace(",}", "}").replace(",]", "]")
    return s

def _looks_like_json(s: str) -> bool:
    s = s.strip()
    return s.startswith("{") and s.endswith("}")

def _extract_balanced_json(text: str) -> Dict[str, Any]:
    """Extract first balanced {...}, respecting quotes/escapes."""
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
                    break
        i += 1

    s = "".join(buf).replace(",}", "}").replace(",]", "]")
    return json.loads(s)

# def parse_agent_output(output_obj: Any) -> Dict[str, Any]:
#     """
#     Accepts whatever LangChain AgentExecutor returned (dict or str) and returns a dict.
#     Priority:
#       1) If dict['output'] is pure JSON -> json.loads
#       2) If string has 'Final Answer:' -> parse JSON after it
#       3) Else -> extract first balanced JSON block
#     """
#     # 1) Dict from AgentExecutor
#     if isinstance(output_obj, dict) and "output" in output_obj:
#         s = str(output_obj["output"]).strip()
#         if _looks_like_json(s):
#             return json.loads(s)
#         output_str = s  # fallthrough
#     else:
#         output_str = str(output_obj)

#     # 2) Try 'Final Answer:' path
#     body = THINK_RE.sub("", output_str)
#     if "Final Answer:" in body:
#         ans = body.split("Final Answer:", 1)[1]
#         ans = _strip_meta(ans)
#         try:
#             return json.loads(ans)
#         except Exception:
#             pass  # fall back to balanced extraction

#     # 3) First balanced JSON
#     return _extract_balanced_json(output_str)


def _find_json_after_marker(text: str, marker: str = "Final Answer:") -> Optional[dict]:
    """
    Return the FIRST balanced JSON object that appears *after* the last occurrence
    of `marker`. If none found or invalid, return None.
    """
    body = THINK_RE.sub("", text or "")
    pos = body.rfind(marker)
    if pos == -1:
        return None
    tail = body[pos + len(marker):]
    tail = _strip_meta(tail)
    try:
        # quick path if it's pure json
        if _looks_like_json(tail):
            return json.loads(tail)
    except Exception:
        pass
    # Robust path: find first balanced {...} in the tail
    try:
        return _extract_balanced_json(tail)
    except Exception:
        return None


def _normalize_meeting_command(cmd: dict) -> dict:
    """
    Light schema cleanup so the gateway gets consistent payloads.
    - ensure objects exist
    - map 'company'/'companies' → 'accounts'
    - dedupe participant_ids and participants
    - trim time HH:MM:SS -> HH:MM
    """
    if not isinstance(cmd, dict):
        return cmd

    action  = cmd.get("action")
    module  = cmd.get("module")

    # Only normalize meetings create
    if (action, module) != ("create", "meetings"):
        return cmd

    params = cmd.setdefault("parameters", {})
    meta   = cmd.setdefault("metadata", {})

    # related_module normalization
    rel_mod = params.get("related_module")
    if rel_mod:
        m = str(rel_mod).lower().strip()
        if m in ("company", "companies"):
            params["related_module"] = "accounts"
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
        # accept "HH:MM" or "HH:MM:SS"
        parts = t.split(":")
        if len(parts) >= 2:
            meta["time"] = f"{parts[0]:0>2}:{parts[1]:0>2}"

    # date: accept "YYYY-MM-DD" (don't hard-fail here)
    d = meta.get("date")
    if isinstance(d, str):
        d = d.strip()
        if d:
            # very light sanity: must have 2 dashes
            if d.count("-") == 2:
                meta["date"] = d

    return cmd


def parse_agent_output(output_obj: Any) -> Dict[str, Any]:
    """
    Accepts whatever LangChain AgentExecutor returned (dict or str) and returns a dict.
    Priority:
      1) If dict['output'] is pure JSON -> json.loads
      2) If there's 'Final Answer:' -> parse the FIRST balanced JSON after the LAST marker
      3) Else -> extract first balanced JSON block from the whole text
    Finally, apply light schema normalization for meetings.
    """
    # 1) Direct dict output (AgentExecutor)
    if isinstance(output_obj, dict) and "output" in output_obj:
        s = str(output_obj["output"]).strip()
        # Some agents return the whole transcript in `output`; try Final Answer first
        j = _find_json_after_marker(s)
        if j is not None:
            return _normalize_meeting_command(j)
        if _looks_like_json(s):
            try:
                return _normalize_meeting_command(json.loads(s))
            except Exception:
                pass
        output_str = s
    else:
        output_str = str(output_obj)

    # 2) Look for Final Answer marker (robust against extra chatter)
    j = _find_json_after_marker(output_str)
    if j is not None:
        return _normalize_meeting_command(j)

    # 3) Fallback: first balanced JSON anywhere in the text
    try:
        j = _extract_balanced_json(output_str)
        return _normalize_meeting_command(j)
    except Exception:
        # Last resort: tell the caller we couldn't parse JSON
        return {"action": "error", "message_to_user": "Omlouvám se, výstup se nepodařilo zpracovat. Zkuste to prosím znovu."}



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

        # Try strict JSON first, then balanced extraction
        try:
            if _looks_like_json(output_text):
                result = json.loads(output_text)
            else:
                result = _extract_balanced_json(output_text)
        except Exception:
            # Nothing JSON-like found
            return {"error": "Model did not return JSON."}

        if add_history:
            chat_history.add_llm_response(result)

        return result

    except Exception as e:
        return {"error": f"Failed to generate JSON command: {str(e)}"}


# ------------------------------- ReAct Agent API ------------------------------

# --- in core/services/llm.py ---
@lru_cache(maxsize=1)
def _react_agent():
    # Generic ReAct wrapper; module-specific schema/instructions come via chat_history
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            # NOTE: no JSON schema or module-specific rules here.
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
        MessagesPlaceholder("chat_history"),  # <- your module agent injects its own system prompt here
        ("human", "{input}"),
        # create_react_agent provides the scratchpad as a STRING:
        ("ai", "{agent_scratchpad}"),
    ])

    agent = create_react_agent(llm=get_chat_llm(), tools=tools, prompt=prompt)

    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=DEBUG_LLM,
        max_iterations=8,
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
        early_stopping_method="generate",
        return_intermediate_steps=False,
    )

def run_react_agent(chat_history: ChatSession) -> Dict[str, Any]:
    try:
        messages = chat_history.to_langchain_messages()

        print(f"🤖 Running ReAct agent with {len(messages)} messages...")

        # last human message becomes the {input}; the rest go into chat_history
        last_user = next((m.content for m in reversed(messages) if m.type == "human"), "")
        output = _react_agent().invoke({
            "chat_history": messages,  # list[BaseMessage]
            "input": last_user,        # str
            # no need to pass agent_scratchpad; AgentExecutor fills it as a string
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
    context_str = ""
    if module_context:
        context_items = [f"{k}: {v}" for k, v in module_context.items()]
        context_str = "CONTEXT:\n" + "\n".join(context_items) + "\n"

    # add context to beginning of system prompt
    if system_prompt:
        system_prompt = f"""{context_str}
        {system_prompt}"""

    chat_history.add_system(system_prompt)

    # Execute shared agent
    result = run_react_agent(chat_history)

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
