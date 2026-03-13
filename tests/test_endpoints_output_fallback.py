from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.api.endpoints import (
    _extract_rag_account_id_from_steps,
    _extract_top_account_id_from_rag_observation,
    _normalize_agent_result_for_ui,
)


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


def test_normalize_agent_result_uses_crm_query_tool_summary() -> None:
    normalized = _normalize_agent_result_for_ui(
        {
            "output": "",
            "intermediate_steps": [
                {
                    "tool": "crm_query_tool",
                    "tool_input": {"module": "Contacts"},
                    "observation": {
                        "status": "ok",
                        "module": "Contacts",
                        "total": 2,
                        "summary": "• Jan Novak\n• Eva Novakova",
                        "cards": [{"id": "table-contacts"}],
                    },
                }
            ],
        }
    )

    assert normalized.get("message_to_user") == "Jan Novak\nEva Novakova"
    assert normalized.get("status") == "ok"
    assert normalized.get("cards") == [{"id": "table-contacts"}]


def test_extract_top_account_id_from_rag_observation_prefers_highest_score() -> None:
    observation = [
        {
            "score": 0.2,
            "payload": {"module": "accounts", "record_id": "11111111-1111-1111-1111-111111111111"},
        },
        {
            "score": 0.96,
            "payload": {"module": "accounts", "record_id": "22222222-2222-2222-2222-222222222222"},
        },
    ]
    assert (
        _extract_top_account_id_from_rag_observation(observation)
        == "22222222-2222-2222-2222-222222222222"
    )


def test_extract_rag_account_id_from_steps_reads_rag_tool_result() -> None:
    steps = [
        {"tool": "crm_query_tool", "observation": {"status": "ok"}},
        {
            "tool": "rag_search_tool",
            "observation": [
                {
                    "score": 0.96,
                    "payload": {
                        "module": "accounts",
                        "record": {"id": "6c3284e1-176c-ed80-77f9-652d0ad83b5c"},
                    },
                }
            ],
        },
    ]
    assert _extract_rag_account_id_from_steps(steps) == "6c3284e1-176c-ed80-77f9-652d0ad83b5c"
