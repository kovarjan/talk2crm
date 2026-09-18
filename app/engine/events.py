from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable


@dataclass
class StreamEvent:
    type: str
    data: dict[str, Any]
    seq: int = 0

    def to_sse(self) -> str:
        return f"event: {self.type}\ndata: {json.dumps(self.data, ensure_ascii=False)}\n\n"


EmitFn = Callable[[StreamEvent], Awaitable[None]]

_TOOL_LABELS: dict[str, Callable[[dict[str, Any]], str]] = {
    "daily_briefing_tool": lambda _: "Připravuji přehled dne…",
    "rag_search_tool":      lambda a: f"Hledám '{a.get('query', '')}'…",
    "crm_query_tool":       lambda a: f"Načítám {a.get('module', '')} z CRM…",
    "crm_action_tool":      lambda _: "Připravuji akci v CRM…",
    "my_meetings_tool":     lambda _: "Načítám schůzky…",
    "get_company_overview": lambda _: "Načítám detail firmy…",
    "crm_aggregate_tool":   lambda _: "Počítám agregaci v CRM…",
    "math_tool":            lambda _: "Počítám výsledek…",
}


def tool_label_cz(tool_name: str, args: dict[str, Any]) -> str:
    fn = _TOOL_LABELS.get(tool_name)
    return fn(args) if fn else f"Zpracovávám {tool_name}…"
