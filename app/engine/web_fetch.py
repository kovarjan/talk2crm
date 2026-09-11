# Copyright 2026 Jan Kovář
# Copyright 2026 Acmark s.r.o.
# SPDX-License-Identifier: Apache-2.0

"""Fetches a single web page and reduces it to readable text.

Companion to web_search.py: web_search_tool finds candidate URLs (title +
short snippet), this lets the agent open one of them to read more than the
snippet. No new third-party HTML parser is pulled in — stdlib's
html.parser is enough to strip markup down to text.

Every request is treated as reaching an untrusted, attacker-influenced
destination (the URL comes from search results or the model itself), so
this module refuses anything that is not a plain public http(s) page:
private/loopback/link-local addresses are blocked before *and* after
following redirects, response size is capped, and the extracted text is
returned as page content only, never as instructions to the agent.
"""

from __future__ import annotations

import ipaddress
import socket
from html.parser import HTMLParser
from urllib.parse import urljoin, urlsplit

from app.core.config import get_settings
from app.core.http import get_shared_http_client
from app.core.logging import get_logger

logger = get_logger(__name__)

_SKIP_TAGS = {"script", "style", "noscript", "template", "svg", "nav", "footer", "header"}
_BLOCK_TAGS = {
    "p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6",
    "section", "article", "table", "ul", "ol", "blockquote",
}
_MAX_REDIRECTS = 5
_MAX_BYTES = 2_000_000


class WebFetchError(Exception):
    """Raised for anything that should surface as a tool-level failure, not a crash."""


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._skip_depth = 0
        self._chunks: list[str] = []
        self.title = ""
        self._in_title = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in _SKIP_TAGS:
            self._skip_depth += 1
        if tag == "title":
            self._in_title = True
        if tag in _BLOCK_TAGS:
            self._chunks.append("\n")

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in _BLOCK_TAGS:
            self._chunks.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in _SKIP_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1
        if tag == "title":
            self._in_title = False
        if tag in _BLOCK_TAGS:
            self._chunks.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        if self._in_title and not self.title:
            self.title = " ".join(data.split())
            return
        self._chunks.append(data)

    def text(self) -> str:
        joined = "".join(self._chunks)
        lines = [" ".join(line.split()) for line in joined.splitlines()]
        return "\n".join(line for line in lines if line)


def _is_public_host(host: str) -> bool:
    """Resolves `host` and rejects it if any address is private/loopback/link-local/reserved."""
    try:
        infos = socket.getaddrinfo(host, None)
    except OSError:
        return False
    for info in infos:
        ip = info[4][0]
        try:
            addr = ipaddress.ip_address(ip)
        except ValueError:
            return False
        if addr.is_private or addr.is_loopback or addr.is_link_local or addr.is_reserved or addr.is_multicast:
            return False
    return True


def _validate_url(url: str) -> str:
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https"):
        raise WebFetchError("Only http/https URLs are supported.")
    if not parts.hostname:
        raise WebFetchError("URL has no host.")
    if not _is_public_host(parts.hostname):
        raise WebFetchError("This host is not reachable (private/internal address).")
    return url


async def fetch_page(*, url: str, max_chars: int | None = None) -> dict[str, str]:
    """Fetches `url` and returns {title, url, text}, `text` capped to `max_chars`.

    Follows up to `_MAX_REDIRECTS` redirects manually, re-validating the
    target host at every hop so a redirect cannot be used to reach an
    internal address.
    """
    settings = get_settings()
    limit = max(500, min(int(max_chars or settings.web_fetch_max_chars), 20_000))
    client = get_shared_http_client()

    current = _validate_url(url)
    response = None
    for _ in range(_MAX_REDIRECTS + 1):
        response = await client.get(
            current,
            headers={"Accept": "text/html,application/xhtml+xml", "User-Agent": "talk2crm-web-fetch/1.0"},
            timeout=settings.web_fetch_timeout_seconds,
            follow_redirects=False,
        )
        if response.status_code in (301, 302, 303, 307, 308) and response.headers.get("location"):
            current = _validate_url(urljoin(current, response.headers["location"]))
            continue
        break
    if response is None:
        raise WebFetchError("Page could not be reached.")
    response.raise_for_status()

    content_type = response.headers.get("content-type", "")
    if "html" not in content_type and "text" not in content_type:
        raise WebFetchError(f"Unsupported content type: {content_type or 'unknown'}")
    if len(response.content) > _MAX_BYTES:
        raise WebFetchError("Page is too large to read.")

    parser = _TextExtractor()
    parser.feed(response.text)
    text = parser.text()[:limit]
    logger.info("web_fetch url=%r chars=%d", current, len(text))
    return {"title": parser.title, "url": current, "text": text}
