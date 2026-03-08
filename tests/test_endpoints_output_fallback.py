from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.api.endpoints import _normalize_agent_result_for_ui


def test_normalize_agent_result_sets_nonempty_fallback_message_for_empty_output() -> None:
    normalized = _normalize_agent_result_for_ui(
        {
            "output": "",
            "intermediate_steps": [
                {
                    "tool": "rag_search_tool",
                    "tool_input": {"module": "Accounts", "query": "Panas"},
                    "observation": '[{"score": 0.96}]',
                }
            ],
        }
    )

    message = str(normalized.get("message_to_user") or "")
    assert message
    assert "Upřesněte prosím akci, modul a čas" in message
