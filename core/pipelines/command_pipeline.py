"""
command_pipeline.py (refactored)

End-to-end flow:
  - (optional) STT -> text
  - ModuleDataExtractor (single-shot, no tools) -> {module, action, parameters}
  - Route by module:
      - meetings -> MeetingsAgent (shared ReAct agent under the hood)
      - contacts -> ContactsAgent (shared ReAct agent under the hood)
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
from core.services.tts import generate_speech
from core.utils.chat import ChatSession
from core.utils.json_validator import validate_json_command

from core.agents.module_data_extractor import ModuleDataExtractor
from core.agents.modules.meetings_agent import MeetingsAgent
from core.agents.modules.contacts_agent import ContactsAgent
from core.agents.modules.tasks_agent import TasksAgent
from core.agents.modules.notes_agent import NotesAgent
from core.agents.modules.calls_agent import CallsAgent
from core.agents.modules.query_agent import QueryAgent

from core.services.llm import query_llm  # kept for generic/fallback chat if needed
import datetime
import os
import logging

# Setup logging
log_dir = "./logs"
os.makedirs(log_dir, exist_ok=True)
log_filename = os.path.join(
    log_dir, f"{datetime.datetime.now().strftime('%Y-%m-%d')}_tools_calls.log"
)
logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)

def _log_timing(t0: float, label: str) -> None:
    if SHOW_TIMING:
        dt = (time.time() - t0) * 1000.0
        print(f"⏱️  {label}: {dt:.1f} ms")


def run_command_pipeline(
    voice_wav_path: Optional[str] = None,
    raw_text: Optional[str] = None,
    history: ChatSession = ChatSession(),
    tenant: str = "unknown",
    extra_context: Optional[Dict[str, Any]] = None,
    user_locale: str = "cs-CZ",
    return_voice: bool = False,
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

    logging.info(f"User input: {text}")

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
        # clarify = query_llm(history, "Parafrázuj požadavek a zeptej se jednou krátkou otázkou pro upřesnění.", returnJson=False, is_assistant_prompt=True)
        # msg = clarify.get("text") if isinstance(clarify, dict) else None
        return {"action": "question", "message_to_user": "Můžete prosím upřesnit, co mám udělat?"}

    module = extraction.get("module")
    action = extraction.get("action")
    params = (extraction.get("parameters") or {})

    history.add_assistant(f"ModuleDataExtractor - user request context: \n"
        f"Module: {module} \n"
        f"Action: {action} \n"
        f"{json.dumps(params, indent=2, ensure_ascii=False)}")

    print(f"🧭  Extracted → module={module}, action={action}, parameters={params}")

    # 4) Route by module
    if action in ("list", "get", "search"):
        t2 = time.time()
        response = QueryAgent(
            command_text=text,
            chat_history=history,
            action=action or "list",
            module=module or "",
            module_context=extra_context or {},
            tenant=tenant,
        )
        _log_timing(t2, "QueryAgent")
    elif module == "meetings":
        t2 = time.time()
        response = MeetingsAgent(
            command_text=text,
            chat_history=history,
            action=action or "",
            module_context=extra_context or {},
            tenant=tenant
        )
        _log_timing(t2, "MeetingsAgent")
    elif module == "contacts":
        t2 = time.time()
        response = ContactsAgent(
            command_text=text,
            chat_history=history,
            action=action or "",
            module_context=extra_context or {},
            tenant=tenant
        )
        _log_timing(t2, "ContactsAgent")
    # elif module in ["tasks", "notes", "calls"]:
    elif module == "tasks":
        t2 = time.time()
        response = TasksAgent(
            command_text=text,
            chat_history=history,
            action=action or "",
            module_context=extra_context or {},
            tenant=tenant,
        )
        _log_timing(t2, "TasksAgent")
    elif module == "notes":
        t2 = time.time()
        response = NotesAgent(
            command_text=text,
            chat_history=history,
            action=action or "",
            module_context=extra_context or {},
            tenant=tenant,
        )
        _log_timing(t2, "NotesAgent")
    elif module == "calls":
        t2 = time.time()
        response = CallsAgent(
            command_text=text,
            chat_history=history,
            action=action or "",
            module_context=extra_context or {},
            tenant=tenant,
        )
        _log_timing(t2, "CallsAgent")
    else:
        # Not implemented yet in this POC
        response = {
            "action": "error",
            "message_to_user": f"Modul '{module}' zatím není implementován v této verzi.",
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

    if return_voice and response.get("message_to_user"):
        t4 = time.time()
        try:
            print(f"🔊 Generating TTS audio response... for message: {response['message_to_user']}")
            audio_id = generate_speech(response["message_to_user"])
            response["audio_response_id"] = audio_id
            _log_timing(t4, "TTS")
        except Exception as e:
            import traceback
            print(f"⚠️  TTS failed: {e}")
            print("⚠️  TTS traceback:\n" + traceback.format_exc())

    print(f"🤖 [Final Response] {json.dumps(response, ensure_ascii=False)}")

    try:
        with open("logs/chat_history.log", "a", encoding="utf-8") as log_file:
            log_file.write(f"Chat History:\n{json.dumps(history.get_messages(), indent=2, ensure_ascii=False)}\n\n")
    except Exception:
        pass

    _log_timing(t0, "Total")
    logging.info(f"Final response: {json.dumps(response, ensure_ascii=False)}")
    logging.info("----------------------------------------")
    logging.info(f"Total processing time: {time.time() - t0:.2f} seconds \n")
    return response
