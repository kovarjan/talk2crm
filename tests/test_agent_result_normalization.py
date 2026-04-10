from __future__ import annotations

from app.api.endpoints import _normalize_agent_result_for_ui


def test_normalization_keeps_agent_final_answer_over_tool_summary() -> None:
    result = _normalize_agent_result_for_ui(
        {
            "output": (
                "Nebyly nalezeny žádné kontakty pro firmu ABClinic s.r.o. "
                "Chcete-li naplánovat schůzku, zadejte prosím jméno kontaktu "
                "nebo potvrďte, že schůzku chcete naplánovat bez konkrétního kontaktu."
            ),
            "intermediate_steps": [
                {
                    "tool": "crm_query_tool",
                    "observation": {
                        "status": "ok",
                        "module": "Contacts",
                        "total": 0,
                        "summary": "Žádné záznamy.",
                        "cards": [],
                    },
                }
            ],
        }
    )

    assert result.get("status") == "ok"
    assert result.get("message_to_user") == result.get("output")


def test_normalization_falls_back_to_tool_summary_when_output_empty() -> None:
    result = _normalize_agent_result_for_ui(
        {
            "output": "",
            "intermediate_steps": [
                {
                    "tool": "crm_query_tool",
                    "observation": {
                        "status": "ok",
                        "summary": "Žádné záznamy.",
                        "cards": [],
                    },
                }
            ],
        }
    )

    assert result.get("message_to_user") == "Žádné záznamy."
