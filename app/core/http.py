# Copyright 2026 Jan Kovář
# Copyright 2026 Acmark s.r.o.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio

import httpx

# Shared pooled HTTP client for outbound calls (CRM, embeddings). Callers are
# constructed per request, so per-instance clients would still open a new
# connection pool every time. The client is bound to the running event loop;
# tests that spin up fresh loops get a fresh client automatically.
_http_client: httpx.AsyncClient | None = None
_http_client_loop: asyncio.AbstractEventLoop | None = None


def get_shared_http_client() -> httpx.AsyncClient:
    global _http_client, _http_client_loop
    loop = asyncio.get_running_loop()
    if _http_client is None or _http_client.is_closed or _http_client_loop is not loop:
        _http_client = httpx.AsyncClient()
        _http_client_loop = loop
    return _http_client


async def aclose_shared_http_client() -> None:
    global _http_client, _http_client_loop
    if _http_client is not None and not _http_client.is_closed:
        await _http_client.aclose()
    _http_client = None
    _http_client_loop = None
