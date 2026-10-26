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
anything that is not a globally routable address (private, loopback,
link-local, CGNAT, reserved) is blocked before *and* after following
redirects, the validated IP is pinned for the connection against DNS
rebinding, response size is capped while streaming, and the extracted text is
returned as page content only, never as instructions to the agent.
"""

from __future__ import annotations

import asyncio
import ipaddress
import socket
from html.parser import HTMLParser
from urllib.parse import urljoin, urlsplit

import httpx

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


def _is_global_ip(ip: str) -> bool:
    try:
        # is_global is False for private, loopback, link-local, CGNAT
        # (100.64/10), documentation, reserved and unspecified ranges.
        return ipaddress.ip_address(ip.split("%", 1)[0]).is_global
    except ValueError:
        return False


async def _resolve_public_ips(host: str) -> list[str] | None:
    """Resolves `host` off the event loop; returns its addresses only if ALL are
    globally routable, else None."""
    try:
        ipaddress.ip_address(host)
    except ValueError:
        pass
    else:
        return [host] if _is_global_ip(host) else None
    try:
        loop = asyncio.get_running_loop()
        infos = await loop.getaddrinfo(host, None, type=socket.SOCK_STREAM)
    except OSError:
        return None
    ips: list[str] = []
    for info in infos:
        ip = str(info[4][0])
        if not _is_global_ip(ip):
            return None
        if ip not in ips:
            ips.append(ip)
    return ips or None


async def _validate_url(url: str) -> tuple[str, str, list[str]]:
    """Returns (url, hostname, validated_ips)."""
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https"):
        raise WebFetchError("Only http/https URLs are supported.")
    host = parts.hostname
    if not host:
        raise WebFetchError("URL has no host.")
    ips = await _resolve_public_ips(host)
    if not ips:
        raise WebFetchError("This host is not reachable (private/internal address).")
    return url, host, ips


def _pin_request(url: str, host: str, ip: str) -> tuple[str, dict[str, str], dict[str, str]]:
    """Rewrites `url` to connect to the already-validated `ip` (defeats DNS
    rebinding between validation and connect) while keeping Host header and
    TLS SNI/certificate verification on the original hostname."""
    parsed = httpx.URL(url)
    if parsed.host != host:
        return url, {}, {}
    pinned = str(parsed.copy_with(host=ip))
    host_header = host if parsed.port is None else f"{host}:{parsed.port}"
    return pinned, {"Host": host_header}, {"sni_hostname": host}


async def fetch_page(*, url: str, max_chars: int | None = None) -> dict[str, str]:
    """Fetches `url` and returns {title, url, text}, `text` capped to `max_chars`.

    Follows up to `_MAX_REDIRECTS` redirects manually, re-validating (and
    re-pinning) the target host at every hop so a redirect cannot be used to
    reach an internal address. The body is streamed and aborted once it
    exceeds `_MAX_BYTES`.
    """
    settings = get_settings()
    limit = max(500, min(int(max_chars or settings.web_fetch_max_chars), 20_000))
    client = get_shared_http_client()

    current, host, ips = await _validate_url(url)
    body = b""
    response_encoding = "utf-8"
    for _ in range(_MAX_REDIRECTS + 1):
        request_url, extra_headers, extensions = _pin_request(current, host, ips[0])
        async with client.stream(
            "GET",
            request_url,
            headers={
                "Accept": "text/html,application/xhtml+xml",
                "User-Agent": "talk2crm-web-fetch/1.0",
                **extra_headers,
            },
            timeout=settings.web_fetch_timeout_seconds,
            follow_redirects=False,
            extensions=extensions,
        ) as response:
            if response.status_code in (301, 302, 303, 307, 308) and response.headers.get("location"):
                current, host, ips = await _validate_url(urljoin(current, response.headers["location"]))
                continue
            response.raise_for_status()
            content_type = response.headers.get("content-type", "")
            if "html" not in content_type and "text" not in content_type:
                raise WebFetchError(f"Unsupported content type: {content_type or 'unknown'}")
            declared = response.headers.get("content-length", "")
            if declared.isdigit() and int(declared) > _MAX_BYTES:
                raise WebFetchError("Page is too large to read.")
            chunks: list[bytes] = []
            received = 0
            async for chunk in response.aiter_bytes():
                received += len(chunk)
                if received > _MAX_BYTES:
                    raise WebFetchError("Page is too large to read.")
                chunks.append(chunk)
            body = b"".join(chunks)
            response_encoding = response.encoding or "utf-8"
            break
    else:
        raise WebFetchError("Too many redirects.")

    parser = _TextExtractor()
    parser.feed(body.decode(response_encoding, errors="replace"))
    text = parser.text()[:limit]
    logger.info("web_fetch url=%r chars=%d", current, len(text))
    return {"title": parser.title, "url": current, "text": text}
