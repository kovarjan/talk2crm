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
from langchain.agents import initialize_agent, AgentType

from core.config import LLM_MODEL_NAME, LLM_TEMPERATURE, DEBUG_LLM, DISABLE_REASONING
from core.utils.chat import ChatSession
from core.utils.json_validator import validate_json_command
from core.services.spellcheck import correct_text
from core.services.tools import tools

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

def parse_agent_output(output_obj: Any) -> Dict[str, Any]:
    """
    Accepts whatever LangChain AgentExecutor returned (dict or str) and returns a dict.
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
        output_str = s  # fallthrough
    else:
        output_str = str(output_obj)

    # 2) Try 'Final Answer:' path
    body = THINK_RE.sub("", output_str)
    if "Final Answer:" in body:
        ans = body.split("Final Answer:", 1)[1]
        ans = _strip_meta(ans)
        try:
            return json.loads(ans)
        except Exception:
            pass  # fall back to balanced extraction

    # 3) First balanced JSON
    return _extract_balanced_json(output_str)


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

@lru_cache(maxsize=1)
def _react_agent():
    # Build a ZERO_SHOT_REACT_DESCRIPTION agent once; it will use the shared chat LLM
    return initialize_agent(
        tools=tools,
        llm=get_chat_llm(),
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=DEBUG_LLM,
        handle_parsing_errors=True,
        max_iterations=6,
    )

def run_react_agent(chat_history: ChatSession) -> Dict[str, Any]:
    """
    Execute a ReAct agent over the chat_history (which must already contain
    the system+user messages for the task). Parses and returns JSON.
    """
    try:
        messages = chat_history.to_langchain_messages()
        output = _react_agent().invoke(messages)
    except Exception as e:
        return {
            "action": "error",
            "message_to_user": "Omlouvám se, došlo k chybě při zpracování požadavku. Zkuste to prosím znovu."
        }

    try:
        result = parse_agent_output(output)
    except Exception:
        return {
            "action": "question",
            "message_to_user": "Omlouvám se, nerozumím přesně zadání. Můžete upřesnit, koho a kdy mám naplánovat?"
        }

    return result


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
