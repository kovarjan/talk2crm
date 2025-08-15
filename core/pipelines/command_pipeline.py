"""
command_pipeline.py (refactored)

End-to-end flow:
  - (optional) STT -> text
  - ModuleDataExtractor (single-shot, no tools) -> {module, action, parameters}
  - Route by module:
      - meetings -> MeetingsAgent (shared ReAct agent under the hood)
      - others -> not implemented (for now)
  - Validate final JSON
  - Return response dict
"""

from __future__ import annotations

import json
import time
from typing import Optional, Dict, Any

from core.config import SHOW_TIMING
from core.services.stt import transcribe_audio
from core.utils.chat import ChatSession
from core.utils.json_validator import validate_json_command

from core.agents.module_data_extractor import ModuleDataExtractor
from core.agents.modules.meetings_agent import MeetingsAgent
from core.services.llm import query_llm  # kept for generic/fallback chat if needed


def _log_timing(t0: float, label: str) -> None:
    if SHOW_TIMING:
        dt = (time.time() - t0) * 1000.0
        print(f"⏱️  {label}: {dt:.1f} ms")


def run_command_pipeline(
    voice_wav_path: Optional[str] = None,
    raw_text: Optional[str] = None,
    history: ChatSession = ChatSession(),
    user_locale: str = "cs-CZ",
    extra_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Main entrypoint: returns a dict with the final action JSON (or an error)."""

    t0 = time.time()

    # 1) Acquire text
    if voice_wav_path:
        text = transcribe_audio(voice_wav_path, language=user_locale) or ""
        _log_timing(t0, "STT")
    else:
        text = (raw_text or "").strip()

    print(f"🎙️  USER: {text}")

    if not text:
        return {"action": "question", "message_to_user": "Neslyšel jsem žádný požadavek. Můžete ho prosím zopakovat?"}

    # 2) New chat history for this request
    response: Dict[str, Any] = {}

    # 3) Module classification (single-shot extractor)
    t1 = time.time()
    moduleDataFromHistory = history.get_module_data()
    if moduleDataFromHistory:
        extraction = moduleDataFromHistory
    else:
        extraction = ModuleDataExtractor(text, chat_history=None)

    _log_timing(t1, "ModuleDataExtractor")

    print(f"🛠️ Extracted:")
    print(extraction)

    if not extraction:
        # Fallback: Ask for clarification via plain chat (no tools, no JSON schema expected)
        print("🧭  Extractor uncertain — asking for clarification.")
        history.add_user(text)
        clarify = query_llm(history, "Parafrázuj požadavek a zeptej se jednou krátkou otázkou pro upřesnění.", returnJson=False, is_assistant_prompt=True)
        msg = clarify.get("text") if isinstance(clarify, dict) else None
        return {"action": "question", "message_to_user": msg or "Můžete prosím upřesnit, co mám udělat?"}

    module = extraction.get("module")
    action = extraction.get("action")
    params = (extraction.get("parameters") or {})

    history.add_assistant(f"ModuleDataExtractor - user request context: \n"
        f"Module: {module} \n"
        f"Action: {action} \n"
        f"{json.dumps(params, indent=2, ensure_ascii=False)}")

    print(f"🧭  Extracted → module={module}, action={action}, parameters={params}")

    # 4) Route by module
    if module == "meetings":
        t2 = time.time()
        response = MeetingsAgent(
            command_text=text,
            chat_history=history,
            action=action or "",
            module_context=extra_context or {},
        )
        _log_timing(t2, "MeetingsAgent")
    else:
        # Not implemented yet in this POC
        response = {
            "action": "error",
            "message_to_user": f"Modul '{module}' zatím není implementován v této ukázce.",
        }

    # 5) Validate final JSON
    t3 = time.time()
    validation = validate_json_command(response)
    _log_timing(t3, "JSON validation")

    if not validation.get("is_valid"):
        print(f"⚠️  Validation failed: {validation}")
        response = {
            "action": "error",
            "message_to_user": "Omlouvám se, formát výsledku není v pořádku. Zkuste to prosím znovu.",
            "details": validation,
        }

    print(f"🤖 [Final Response] {json.dumps(response, ensure_ascii=False)}")

    try:
        with open("logs/chat_history.log", "a", encoding="utf-8") as log_file:
            log_file.write(f"Chat History:\n{json.dumps(history.get_messages(), indent=2, ensure_ascii=False)}\n\n")
    except Exception:
        pass

    _log_timing(t0, "Total")
    return response
