# Copyright 2026 Jan Kovář
# Copyright 2026 Acmark s.r.o.
# SPDX-License-Identifier: Apache-2.0

"""Date-range parsing and Coripo date filters for meeting queries."""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from typing import Any

from app.presentation.cards import parse_datetime
from app.utils.text import safe_text

def parse_date_token(value: str) -> datetime | None:
    raw = safe_text(value)
    if not raw:
        return None

    for fmt in ("%Y-%m-%d", "%d.%m.%Y", "%d.%m.%y"):
        try:
            return datetime.strptime(raw, fmt)
        except ValueError:
            continue
    return None

def extract_date_range_from_text(text: str) -> tuple[datetime, datetime] | None:
    source = safe_text(text)
    if not source:
        return None

    tokens = re.findall(r"\d{4}-\d{2}-\d{2}|\d{1,2}\.\d{1,2}\.\d{2,4}", source)
    parsed: list[datetime] = []
    for token in tokens:
        dt = parse_date_token(token)
        if dt is not None:
            parsed.append(dt.replace(hour=0, minute=0, second=0, microsecond=0))

    if len(parsed) < 2:
        return None
    start = min(parsed)
    end = max(parsed) + timedelta(days=1)
    return start, end

def next_week_range(now: datetime | None = None) -> tuple[datetime, datetime]:
    current = now or datetime.now()
    days_to_next_monday = 7 - current.weekday()
    start = (current + timedelta(days=days_to_next_monday)).replace(
        hour=0,
        minute=0,
        second=0,
        microsecond=0,
    )
    end = start + timedelta(days=7)
    return start, end

def build_meetings_date_filter(range_start: datetime, range_end_exclusive: datetime) -> dict[str, Any]:
    end_inclusive = (range_end_exclusive - timedelta(days=1)).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    return {
        "operator": "and",
        "operands": [
            {
                "operator": "and",
                "operands": [
                    {
                        "field": "date_start",
                        "fieldModule": None,
                        "fieldRel": None,
                        "type": "moreThanInclude",
                        "value": range_start.strftime("%Y-%m-%d"),
                        "relationField": None,
                    },
                    {
                        "field": "date_start",
                        "fieldModule": None,
                        "fieldRel": None,
                        "type": "lessThanInclude",
                        "value": end_inclusive.strftime("%Y-%m-%d"),
                        "relationField": None,
                    },
                ],
            }
        ],
    }

def format_filter_datetime_boundary(value: str) -> str:
    raw = safe_text(value)
    dt = parse_datetime(raw)
    if dt is None:
        return ""
    if re.search(r"\d{1,2}:\d{2}", raw):
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    return dt.strftime("%Y-%m-%d")

def build_login_user_meetings_window_filter(
    date_from: str | None,
    date_to: str | None,
) -> dict[str, Any]:
    operands: list[dict[str, Any]] = [
        {
            "field": "assigned_user_id",
            "type": "eq",
            "value": "{%LOGIN_USER%}",
        }
    ]

    if date_from:
        operands.append(
            {
                "field": "date_start",
                "fieldModule": None,
                "fieldRel": None,
                "type": "moreThanInclude",
                "value": date_from,
                "relationField": None,
            }
        )
    if date_to:
        operands.append(
            {
                "field": "date_start",
                "fieldModule": None,
                "fieldRel": None,
                "type": "lessThanInclude",
                "value": date_to,
                "relationField": None,
            }
        )

    return {"operator": "and", "operands": operands}

def sort_records_by_datetime(records: list[dict[str, Any]], field_name: str) -> list[dict[str, Any]]:
    return sorted(
        records,
        key=lambda row: parse_datetime(row.get(field_name)) or datetime.max,
    )
