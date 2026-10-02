from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict

from app.utils.text import safe_text as _safe_text

if TYPE_CHECKING:  # pragma: no cover
    from app.services.module_catalog import ModuleCatalog

_RAG_MODULES = {
    "",
    "contacts",
    "accounts",
    "meetings",
    "calls",
    "tasks",
    "notes",
    "leads",
    "opportunities",
    "opportunites",
    "quotes",
    "acm_invoices",
    "producttemplates",
}

class RagSearchToolArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str
    limit: int = 5
    module: str = ""


class WebSearchToolArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str
    max_results: int = 5


def validate_web_search_call(*, query: str, max_results: int) -> str | None:
    if not _safe_text(query):
        return "Query is required."
    if int(max_results) < 1 or int(max_results) > 20:
        return "max_results must be between 1 and 20."
    return None


class WebFetchToolArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    url: str


def validate_web_fetch_call(*, url: str) -> str | None:
    value = _safe_text(url)
    if not value:
        return "URL is required."
    if not (value.startswith("http://") or value.startswith("https://")):
        return "Only http/https URLs are supported."
    return None


class CrmAggregateToolArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    module: str
    operation: str
    metric: str = "amount"
    account_id: str | None = None
    date_from: str | None = None
    date_to: str | None = None
    date_field: str = "date_issued"
    statuses: list[str] | None = None
    exclude_cancelled: bool = True
    assigned_user_id: str | None = None


class MathToolArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    numbers: list[float]
    operation: str


def validate_rag_search_call(*, query: str, limit: int, module: str, catalog: "ModuleCatalog | None" = None) -> str | None:
    if not _safe_text(query):
        return "Query is required."
    if int(limit) < 1 or int(limit) > 100:
        return "Limit must be between 1 and 100."
    allowed_rag = (catalog.rag() | {""}) if catalog is not None else _RAG_MODULES
    if _safe_text(module).lower() not in allowed_rag:
        return f"Unsupported module filter '{module}'."
    return None


