"""Shared fakes for the AI tool registry tests (not a test module)."""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Any

import httpx

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Written by rest_coripo's ManifestContractTest (site/tests/fixtures/); copy it here when
# the CRM tool definitions change so both sides test against the same protocol.
SAMPLE_MANIFEST_PATH = Path(__file__).parent / "fixtures" / "ai_tools_manifest.sample.json"


def sample_manifest() -> dict[str, Any]:
    return json.loads(SAMPLE_MANIFEST_PATH.read_text(encoding="utf-8"))


def ok_result(structured: dict[str, Any], text: str = "ok", meta: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": text}], "structuredContent": structured, "isError": False, "meta": meta or {}}


def error_result(code: str, message: str, details: list | None = None) -> dict[str, Any]:
    return {
        "content": [{"type": "text", "text": message}],
        "structuredContent": {"status": "error", "error": {"code": code, "message": message, "details": details or []}},
        "isError": True,
        "meta": {},
    }


def http_error(status: int) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "http://crm/ai/tools/call")
    return httpx.HTTPStatusError(f"{status}", request=request, response=httpx.Response(status, request=request))


class FakeTransport:
    """Scripted ToolTransport: `responses` is a list of dicts or exceptions, consumed per call."""

    def __init__(self, manifest: dict[str, Any] | None = None, responses: list[Any] | None = None) -> None:
        self.manifest = copy.deepcopy(manifest) if manifest is not None else sample_manifest()
        self.responses = list(responses or [])
        self.calls: list[dict[str, Any]] = []
        self.manifest_calls: list[str | None] = []
        self.manifest_error: Exception | None = None

    async def fetch_manifest(self, etag: str | None) -> tuple[bool, dict[str, Any] | None]:
        self.manifest_calls.append(etag)
        if self.manifest_error is not None:
            raise self.manifest_error
        if etag and etag == self.manifest.get("manifest_hash"):
            return True, None
        return False, copy.deepcopy(self.manifest)

    async def call(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        response = self.responses.pop(0) if self.responses else ok_result({"status": "ok"})
        if isinstance(response, BaseException):
            raise response
        return response


class FakeNativeTool:
    def __init__(self, name: str, output: str = '{"status": "ok"}') -> None:
        self.name = name
        self.output = output
        self.calls: list[dict[str, Any]] = []

    async def ainvoke(self, args: dict[str, Any]) -> str:
        self.calls.append(args)
        return self.output


class MemorySnapshots:
    def __init__(self) -> None:
        self.rows: dict[str, tuple[dict[str, Any], float]] = {}
        self.saves = 0

    async def load(self, tenant_id: str) -> tuple[dict[str, Any], float] | None:
        return self.rows.get(tenant_id)

    async def save(self, tenant_id: str, raw: dict[str, Any]) -> None:
        import time

        self.saves += 1
        self.rows[tenant_id] = (copy.deepcopy(raw), time.time())
