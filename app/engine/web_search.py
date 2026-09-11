# Copyright 2026 Jan Kovář
# Copyright 2026 Acmark s.r.o.
# SPDX-License-Identifier: Apache-2.0

"""Web search through a self-hosted SearXNG instance.

SearXNG runs as the `searxng` service in the docker-compose stack and must have
`search.formats` including `json` (see searxng/settings.yml). The gateway only
ever talks to its own instance, never to public search engines directly.
"""

from __future__ import annotations

from typing import Any

from app.core.config import get_settings
from app.core.http import get_shared_http_client
from app.core.logging import get_logger

logger = get_logger(__name__)


def _clean(value: Any, limit: int) -> str:
    text = " ".join(str(value or "").split())
    return text[:limit]


def normalize_searxng_results(payload: dict[str, Any], max_results: int) -> list[dict[str, str]]:
    """Reduces a SearXNG JSON response to {title, url, content} rows, deduplicated by url."""
    rows = payload.get("results") if isinstance(payload, dict) else None
    out: list[dict[str, str]] = []
    seen: set[str] = set()
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        url = _clean(row.get("url"), 512)
        if not url or url in seen:
            continue
        seen.add(url)
        out.append(
            {
                "title": _clean(row.get("title"), 200),
                "url": url,
                "content": _clean(row.get("content"), 600),
            }
        )
        if len(out) >= max_results:
            break
    return out


async def search_web(*, query: str, max_results: int | None = None) -> list[dict[str, str]]:
    settings = get_settings()
    limit = max(1, min(int(max_results or settings.web_search_max_results), 20))
    url = f"{settings.searxng_url.rstrip('/')}/search"
    response = await get_shared_http_client().get(
        url,
        params={"q": query, "format": "json", "language": "cs", "safesearch": 1},
        headers={"Accept": "application/json"},
        timeout=settings.web_search_timeout_seconds,
    )
    response.raise_for_status()
    payload = response.json()
    results = normalize_searxng_results(payload, limit)
    logger.info("web_search query=%r results=%d", query, len(results))
    return results
