from __future__ import annotations

import re
import unicodedata
from typing import Any

from app.domain.contracts import NormalizedCommand


_EMAIL_RE = re.compile(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", re.IGNORECASE)
_PHONE_RE = re.compile(r"(?:\+420\s*)?(\d[\d\s]{7,}\d)")


class CommandParser:
    @staticmethod
    def _normalize_text(value: str) -> str:
        lowered = (value or "").strip().lower()
        unaccented = "".join(
            c for c in unicodedata.normalize("NFD", lowered) if unicodedata.category(c) != "Mn"
        )
        return re.sub(r"[^a-z0-9]+", " ", unaccented).strip()

    def parse(self, text: str) -> NormalizedCommand | None:
        contact_payload = self._extract_create_contact_payload(text)
        if contact_payload is not None:
            return NormalizedCommand(
                intent="create_contact",
                module="Contacts",
                action="create",
                slots=contact_payload,
                confidence=0.95,
                source="quick_action_regex",
            )

        meeting_payload = self._extract_create_meeting_payload(text)
        if meeting_payload is not None:
            return NormalizedCommand(
                intent="create_meeting",
                module="Meetings",
                action="create",
                slots=meeting_payload,
                confidence=0.95,
                source="quick_action_regex",
            )

        normalized = self._normalize_text(text)
        if self._is_latest_leads_query(normalized):
            return NormalizedCommand(
                intent="latest_leads",
                module="Leads",
                action="list",
                slots={},
                confidence=0.9,
                source="quick_action_regex",
            )
        if self._is_week_meetings_query(normalized):
            return NormalizedCommand(
                intent="week_meetings",
                module="Meetings",
                action="list",
                slots={},
                confidence=0.9,
                source="quick_action_regex",
            )
        return None

    @staticmethod
    def _is_latest_leads_query(normalized: str) -> bool:
        if "zajemc" not in normalized and "lead" not in normalized:
            return False
        return any(trigger in normalized for trigger in ("nejnovejs", "nove", "posledn"))

    @staticmethod
    def _is_week_meetings_query(normalized: str) -> bool:
        if "tento tyden" not in normalized and "tenhle tyden" not in normalized:
            return False
        return any(token in normalized for token in ("schuzk", "scuzk", "plan", "meeting"))

    def _extract_create_contact_payload(self, text: str) -> dict[str, Any] | None:
        normalized = self._normalize_text(text)
        if "kontakt" not in normalized:
            return None
        if not any(trigger in normalized for trigger in ("vytvor", "zaloz", "pridej kontakt", "novy kontakt")):
            return None

        name_match = re.search(
            r"vytvo[rř]\s+kontakt\s+(.+?)(?=\s+(?:tel|telefon|mail|email|e-mail|pridej|přidej|ke\s+spole[cč]nosti)\b|$)",
            text,
            re.IGNORECASE,
        )
        if not name_match:
            name_match = re.search(
                r"zalo[zž]\s+kontakt\s+(.+?)(?=\s+(?:tel|telefon|mail|email|e-mail|pridej|přidej|ke\s+spole[cč]nosti)\b|$)",
                text,
                re.IGNORECASE,
            )

        full_name = ""
        if name_match:
            full_name = re.sub(r"\s+", " ", name_match.group(1)).strip(" ,.;")

        if not full_name:
            return None

        parts = full_name.split()
        first_name = parts[0]
        last_name = " ".join(parts[1:]) if len(parts) > 1 else ""

        email_match = _EMAIL_RE.search(text)
        phone_match = _PHONE_RE.search(text)
        company_match = re.search(
            r"(?:ke\s+spole[cč]nosti|k\s+firm[eě]|do\s+spole[cč]nosti)\s+([^,.;:]+)",
            text,
            re.IGNORECASE,
        )

        payload: dict[str, Any] = {
            "first_name": first_name,
            "last_name": last_name,
        }
        if email_match:
            payload["email1"] = email_match.group(0).strip()
        if phone_match:
            payload["phone_mobile"] = re.sub(r"\s+", "", phone_match.group(1))
        if company_match:
            payload["company_name_hint"] = company_match.group(1).strip(" ,.;")
        return payload

    def _extract_create_meeting_payload(self, text: str) -> dict[str, Any] | None:
        normalized = self._normalize_text(text)
        if "schuzk" not in normalized and "meeting" not in normalized:
            return None
        if not any(
            trigger in normalized
            for trigger in ("vytvor", "zaloz", "naplanuj", "pridej", "domluv", "zarid")
        ):
            return None

        person_match = re.search(
            (
                r"\b(?:s\s+)?(?:pan[ií]|panem|pan)\s+([^,.;:]+?)"
                r"(?=\s+(?:z|ze|na|v|ve|o|ohledn[eě]|kv[uů]li|k|do|od|u)\b|$)"
            ),
            text,
            re.IGNORECASE,
        )
        if not person_match:
            person_match = re.search(
                r"\b(?:s|se)\s+([^,.;:]+?)(?=\s+(?:z|ze|na|v|ve|o|ohledn[eě]|kv[uů]li|k|do|od|u)\b|$)",
                text,
                re.IGNORECASE,
            )
        company_match = re.search(
            (
                r"\b(?:z|ze|od)\s+(?:firmy|spole[cč]nosti|company)\s+([^,.;:]+?)"
                r"(?=\s+(?:na|v|ve|o|ohledn[eě]|kv[uů]li|k|do|od|u|s|se)\b|$)"
            ),
            text,
            re.IGNORECASE,
        )
        topic_match = re.search(
            r"\b(?:ohledn[eě]|kv[uů]li|pro)\s+([^,.;:]+)",
            text,
            re.IGNORECASE,
        )

        fields: dict[str, Any] = {}
        if person_match:
            contact_hint = re.sub(r"\s+", " ", person_match.group(1)).strip(" ,.;")
            contact_hint_norm = self._normalize_text(contact_hint)
            if contact_hint_norm and not any(
                token in contact_hint_norm for token in ("firma", "firm", "spolecnost", "company")
            ):
                fields["contact_name"] = contact_hint
        if company_match:
            fields["account_name"] = re.sub(r"\s+", " ", company_match.group(1)).strip(" ,.;")
        if topic_match:
            fields["description"] = re.sub(r"\s+", " ", topic_match.group(1)).strip(" ,.;")

        return {"fields": fields}
