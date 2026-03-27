from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timedelta
from difflib import SequenceMatcher
from typing import Any

from app.domain.contracts import TemporalResolution
from app.domain.resolver_policy import ResolverPolicy, load_resolver_policy


_TIME_RE = re.compile(r"\b(?P<hour>\d{1,2})[:.](?P<minute>\d{2})\b")


@dataclass
class TemporalResolver:
    policy: ResolverPolicy

    @classmethod
    def from_settings(cls) -> "TemporalResolver":
        return cls(policy=load_resolver_policy())

    @staticmethod
    def _normalize_text(value: str) -> str:
        lowered = (value or "").strip().lower()
        unaccented = "".join(
            c for c in unicodedata.normalize("NFD", lowered) if unicodedata.category(c) != "Mn"
        )
        return re.sub(r"[^a-z0-9]+", " ", unaccented).strip()

    @classmethod
    def _weekday_from_text(cls, text: str) -> int | None:
        normalized = cls._normalize_text(text)
        mapping = {
            "pondeli": 0,
            "utery": 1,
            "streda": 2,
            "ctvrtek": 3,
            "patek": 4,
            "sobota": 5,
            "nedele": 6,
        }
        for key, value in mapping.items():
            if key in normalized:
                return value

        tokens = [token for token in normalized.split() if token]
        for token in tokens:
            for key, value in mapping.items():
                if len(token) < 4 or abs(len(token) - len(key)) > 2:
                    continue
                if SequenceMatcher(None, token, key).ratio() >= 0.8:
                    return value
        return None

    @staticmethod
    def _first_nonempty(data: dict[str, Any], keys: list[str]) -> str | None:
        for key in keys:
            value = data.get(key)
            if value is None:
                continue
            text = str(value).strip()
            if text:
                return text
        return None

    @staticmethod
    def _parse_datetime(value: str) -> datetime | None:
        raw = str(value or "").strip()
        if not raw:
            return None
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
            try:
                return datetime.strptime(raw, fmt)
            except ValueError:
                continue
        try:
            return datetime.fromisoformat(raw.replace("Z", "+00:00")).replace(tzinfo=None)
        except Exception:
            return None

    def resolve_meeting_datetime(
        self,
        *,
        fields: dict[str, Any],
        input_text: str,
        now: datetime | None = None,
    ) -> TemporalResolution:
        current = self._first_nonempty(fields, ["date_start", "scheduled_time", "scheduled_at", "start_time"])
        parsed_current = self._parse_datetime(current or "") if current else None

        pivot = now or datetime.now()
        if parsed_current and parsed_current.year >= pivot.year - 1:
            return TemporalResolution(
                status="resolved",
                datetime_value=parsed_current.strftime("%Y-%m-%d %H:%M:%S"),
                inferred_fields=[],
                confidence=0.99,
                needs_confirmation=False,
            )

        weekday = self._weekday_from_text(input_text)
        time_match = _TIME_RE.search(input_text or "")
        inferred_fields: list[str] = []
        hour = 9
        minute = 0
        if time_match:
            hour = max(0, min(23, int(time_match.group("hour"))))
            minute = max(0, min(59, int(time_match.group("minute"))))
        else:
            normalized_input = self._normalize_text(input_text or "")
            if "po poledni" in normalized_input:
                hour, minute = 13, 0
                inferred_fields.append("part_of_day")
            elif "odpoledne" in normalized_input:
                hour, minute = 14, 0
                inferred_fields.append("part_of_day")
            elif "hned rano" in normalized_input or "rano" in normalized_input:
                hour, minute = 8, 0
                inferred_fields.append("part_of_day")

        if weekday is None:
            if parsed_current:
                return TemporalResolution(
                    status="resolved",
                    datetime_value=parsed_current.strftime("%Y-%m-%d %H:%M:%S"),
                    inferred_fields=inferred_fields,
                    confidence=0.9,
                    needs_confirmation=False,
                )
            return TemporalResolution(
                status="unresolved",
                datetime_value=None,
                inferred_fields=inferred_fields,
                confidence=0.0,
                needs_confirmation=False,
            )

        delta = (weekday - pivot.weekday()) % 7
        if delta == 0 and (hour, minute) <= (pivot.hour, pivot.minute):
            delta = 7

        target = (pivot + timedelta(days=delta)).replace(
            hour=hour,
            minute=minute,
            second=0,
            microsecond=0,
        )
        inferred_fields.append("weekday")

        needs_confirmation = bool(inferred_fields)
        return TemporalResolution(
            status="resolved",
            datetime_value=target.strftime("%Y-%m-%d %H:%M:%S"),
            inferred_fields=inferred_fields,
            confidence=0.74 if needs_confirmation else 0.95,
            needs_confirmation=needs_confirmation,
        )
