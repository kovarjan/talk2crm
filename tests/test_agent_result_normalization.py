from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.presentation.agent_result import normalize_agent_result_for_ui as _normalize_agent_result_for_ui


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


def test_normalization_strips_answer_tag_from_user_message() -> None:
    result = _normalize_agent_result_for_ui(
        {
            "output": "<answer>Našel jsem příležitost ELEMAN - CRM CORIPO.",
            "intermediate_steps": [],
        }
    )

    assert result.get("message_to_user") == "Našel jsem příležitost ELEMAN - CRM CORIPO."


def test_normalization_formats_iso_dates_for_user_message() -> None:
    result = _normalize_agent_result_for_ui(
        {
            "output": (
                "Datum uzavření: 2021-11-08. "
                "Poslední faktura z 2026-01-31. "
                "Změněno 2026-05-16 10:21:00."
            ),
            "intermediate_steps": [],
        }
    )

    assert result.get("message_to_user") == (
        "Datum uzavření: 8.11.2021. "
        "Poslední faktura z 31.1.2026. "
        "Změněno 16.5.2026 10:21."
    )
