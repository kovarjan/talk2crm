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


def test_long_analytical_answer_survives_over_tool_summary() -> None:
    # Regression: answers over 1400 chars were flagged as "noise" and replaced
    # by the raw tool summary (bullet list of record names).
    long_answer = (
        "Našel jsem jen nabídky, ne objednávky ani obchodní případy. "
        + "Dlouhodobě dominují vidlice a prodloužení vidlic, opakovaně se objevují speciální varianty. "
        * 30
    )
    assert len(long_answer) > 1400
    result = _normalize_agent_result_for_ui(
        {
            "output": long_answer,
            "intermediate_steps": [
                {
                    "tool": "crm_query_tool",
                    "observation": {
                        "status": "ok",
                        "module": "Quotes",
                        "total": 51,
                        "summary": "• Linde\n• prodloužení\n• vidlice",
                        "cards": [],
                    },
                }
            ],
        }
    )

    assert result.get("message_to_user", "").startswith("Našel jsem jen nabídky")


def test_long_answer_after_company_overview_not_replaced_by_fallback() -> None:
    # Regression: get_company_overview is not a card-producing data tool, so the
    # result fell through to the tail where long answers became the generic
    # "Nerozuměla jsem..." fallback.
    long_answer = (
        "Našel jsem jen omezený přehled firmy, ne detail objednávek po položkách. "
        + "V roce 2026 je v aktivitách víc zmínek o SolidAir a nových VZV Linde. " * 30
    )
    assert len(long_answer) > 1400
    result = _normalize_agent_result_for_ui(
        {
            "output": long_answer,
            "intermediate_steps": [
                {
                    "tool": "get_company_overview",
                    "observation": {"module": "Accounts", "record": {"name": "PRO-DOMA, SE"}},
                }
            ],
        }
    )

    message = result.get("message_to_user", "")
    assert "Nerozuměla jsem" not in message
    assert message.startswith("Našel jsem jen omezený přehled firmy")


def test_fallback_message_is_action_neutral() -> None:
    result = _normalize_agent_result_for_ui({"output": "", "intermediate_steps": []})
    message = result.get("message_to_user", "")
    assert message
    assert "schůzka" not in message
    assert "modul" not in message


def test_json_blob_output_still_treated_as_noise() -> None:
    blob = '{"menu": ' + '{"row_count": 5, "items": ["a", "b"]}' * 40 + "}"
    result = _normalize_agent_result_for_ui(
        {
            "output": blob,
            "intermediate_steps": [
                {
                    "tool": "crm_query_tool",
                    "observation": {"status": "ok", "summary": "Žádné záznamy.", "cards": []},
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


def test_confirmation_message_describes_pending_action():
    from app.presentation.agent_result import normalize_agent_result_for_ui

    agent_result = {
        "output": "",
        "intermediate_steps": [
            {
                "tool": "crm_action_tool",
                "observation": {
                    "status": "confirmation_required",
                    "message": "Akce mění CRM data a vyžaduje potvrzení uživatele.",
                    "pending_action": {
                        "module": "Notes",
                        "action": "create",
                        "data": {
                            "fields": {
                                "note_content": "Tajný kód: Ax554Cpm43",
                                "parent_name": "ELEMAN spol. s r.o.",
                            }
                        },
                        "requires_confirmation": True,
                    },
                },
            }
        ],
    }
    result = normalize_agent_result_for_ui(agent_result)
    msg = result.get("message_to_user") or ""
    # The persisted assistant message must describe WHAT is pending,
    # not just the generic confirmation sentence.
    assert "Tajný kód: Ax554Cpm43" in msg
    assert "poznámka" in msg
    assert "potvrzení" in msg.lower()


def test_describe_pending_action_cz_meeting():
    from app.presentation.agent_result import describe_pending_action_cz

    msg = describe_pending_action_cz(
        "Meetings",
        "create",
        {"fields": {"name": "Schůzka s Iveco", "date_start": "2026-07-04 11:00", "contact_name": "Jan Kovář"}},
    )
    assert "vytvoření" in msg
    assert "schůzka" in msg
    assert "Schůzka s Iveco" in msg
    assert "2026-07-04 11:00" in msg
