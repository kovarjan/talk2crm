#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import hashlib
import hmac
import json
import os
import re
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib import error, parse, request


class ValidationError(RuntimeError):
    pass


@dataclass
class RecordRow:
    record_id: str
    name: str
    date_modified: str


class JsonHttpClient:
    def __init__(self, timeout: float = 60.0):
        self.timeout = timeout

    def request_json(
        self,
        method: str,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        body_bytes: bytes | None = None
        req_headers = dict(headers or {})
        if body is not None:
            body_bytes = _json_dumps(body)
            req_headers.setdefault("Content-Type", "application/json")

        req = request.Request(url=url, method=method.upper(), headers=req_headers, data=body_bytes)
        try:
            with request.urlopen(req, timeout=self.timeout) as response:
                payload = response.read().decode("utf-8", errors="replace")
        except error.HTTPError as exc:
            text = exc.read().decode("utf-8", errors="replace")
            raise ValidationError(f"HTTP {exc.code} for {method} {url}: {text[:500]}") from exc
        except error.URLError as exc:
            raise ValidationError(f"Request failed for {method} {url}: {exc}") from exc

        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ValidationError(f"Non-JSON response for {method} {url}: {payload[:500]}") from exc

        if not isinstance(parsed, dict):
            raise ValidationError(f"Unexpected JSON shape for {method} {url}: {type(parsed).__name__}")
        return parsed


def _json_dumps(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str).encode("utf-8")


def _parse_dotenv(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values

    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip()
        if not key:
            continue
        val = val.strip()
        if (val.startswith("\"") and val.endswith("\"")) or (val.startswith("'") and val.endswith("'")):
            val = val[1:-1]
        values[key] = val
    return values


_ENV_VAR_RE = re.compile(r"\$(\{[A-Za-z_][A-Za-z0-9_]*\}|[A-Za-z_][A-Za-z0-9_]*)")


def _expand_env_vars(value: str, env_map: dict[str, str]) -> str:
    def repl(match: re.Match[str]) -> str:
        token = match.group(1)
        name = token[1:-1] if token.startswith("{") else token
        return env_map.get(name, "")

    return _ENV_VAR_RE.sub(repl, value)


def _sanitize_collection_segment(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", str(value or "").strip())
    cleaned = cleaned.strip("_")
    return cleaned or "tenant"


def _tenant_collection(prefix: str, tenant_id: str) -> str:
    pfx = _sanitize_collection_segment(prefix)
    tenant = _sanitize_collection_segment(tenant_id)
    candidate = f"{pfx}__{tenant}"
    if len(candidate) > 190:
        suffix = hashlib.sha1(tenant.encode("utf-8")).hexdigest()[:10]
        candidate = f"{pfx}__{tenant[:150]}__{suffix}"
    return candidate


def _build_hmac_headers(*, key_id: str, secret: str, user_id: str, user_name: str | None, body_bytes: bytes) -> dict[str, str]:
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    nonce = str(uuid.uuid4())
    sign_base = f"{timestamp}|{nonce}|{user_id}|".encode("utf-8") + body_bytes
    signature = base64.b64encode(hmac.new(secret.encode("utf-8"), sign_base, hashlib.sha256).digest()).decode("utf-8")

    headers = {
        "Authorization": f"HMAC keyId={key_id}, signature={signature}",
        "X-Timestamp": timestamp,
        "X-Nonce": nonce,
        "X-User-Id": user_id,
    }
    if user_name:
        headers["X-User-Name"] = user_name
    return headers


def _pick_first(*values: str | None) -> str | None:
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return None


def _to_record_name(record: dict[str, Any]) -> str:
    for key in ("name", "full_name", "subject", "title", "document_name"):
        value = str(record.get(key) or "").strip()
        if value:
            return value

    first = str(record.get("first_name") or "").strip()
    last = str(record.get("last_name") or "").strip()
    full = f"{first} {last}".strip()
    return full or "<unknown>"


def _extract_sid(payload: dict[str, Any]) -> str | None:
    for key in ("sid", "session_id"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    message = payload.get("message")
    if isinstance(message, dict):
        for key in ("sid", "session_id"):
            value = message.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        data = message.get("data")
        if isinstance(data, dict):
            for key in ("sid", "session_id"):
                value = data.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()

    data = payload.get("data")
    if isinstance(data, dict):
        for key in ("sid", "session_id"):
            value = data.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

    return None


def _build_runtime_defaults(args: argparse.Namespace) -> dict[str, str]:
    root = Path(__file__).resolve().parents[1]
    app_env = _parse_dotenv(root / ".env")

    crm_env: dict[str, str] = {}
    crm_path = Path(args.crm_path).expanduser().resolve() if args.crm_path else None
    if crm_path:
        crm_env = _parse_dotenv(crm_path / ".env")

    merged = dict(app_env)
    merged.update(crm_env)
    merged.update({k: v for k, v in os.environ.items() if v is not None})

    backend_url_raw = _pick_first(
        args.crm_base_url,
        merged.get("CORIPO_TEST_BASE_URL"),
        merged.get("BACKEND_URL"),
        "http://localhost:2000/public",
    )
    if backend_url_raw is None:
        raise ValidationError("CRM base URL cannot be resolved")

    backend_url = _expand_env_vars(backend_url_raw, merged).rstrip("/")
    if not backend_url.endswith("/public"):
        backend_url = f"{backend_url}/public"

    qdrant_url = _pick_first(args.qdrant_url, merged.get("QDRANT_URL"), "http://localhost:6333")
    if qdrant_url is None:
        raise ValidationError("Qdrant URL cannot be resolved")
    qdrant_api_key = _pick_first(
        args.qdrant_api_key,
        merged.get("QDRANT_API_KEY"),
        merged.get("QDRANT__SERVICE__API_KEY"),
    )

    hmac_key_id = _pick_first(args.hmac_key_id, merged.get("CORIPO_HMAC_KEY_ID"), merged.get("AI_GATEWAY_HMAC_KEY_ID"), "your-tenant-id")
    hmac_secret = _pick_first(args.hmac_secret, merged.get("CORIPO_TEST_TOKEN"), merged.get("AI_GATEWAY_HMAC_SECRET"), merged.get("CORIPO_TEST_HMAC_SECRET"))
    user_id = _pick_first(args.user_id, merged.get("CORIPO_TEST_USER_ID"), merged.get("CRM_TEST_USER_ID"), "1")
    user_name = _pick_first(args.user_name, merged.get("CORIPO_TEST_USER_NAME"), merged.get("CRM_TEST_USER_NAME"), "testuser")

    collection = _pick_first(args.collection)
    if collection is None:
        prefix = _pick_first(args.collection_prefix, merged.get("QDRANT_COLLECTION"), "tenant_knowledge")
        if prefix is None:
            raise ValidationError("Qdrant collection prefix cannot be resolved")
        collection = _tenant_collection(prefix, args.tenant_id)

    return {
        "crm_base_url": backend_url,
        "qdrant_url": qdrant_url.rstrip("/"),
        "qdrant_api_key": qdrant_api_key or "",
        "hmac_key_id": hmac_key_id or "",
        "hmac_secret": hmac_secret or "",
        "user_id": user_id or "",
        "user_name": user_name or "",
        "collection": collection,
    }


def _login_coripo(
    client: JsonHttpClient,
    *,
    crm_base_url: str,
    hmac_key_id: str,
    hmac_secret: str,
    user_id: str,
    user_name: str,
) -> str:
    if not hmac_secret:
        raise ValidationError("Missing HMAC secret and --sid not provided")
    if not user_id:
        raise ValidationError("Missing user id and --sid not provided")

    body = {}
    body_bytes = _json_dumps(body)
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    headers.update(
        _build_hmac_headers(
            key_id=hmac_key_id,
            secret=hmac_secret,
            user_id=user_id,
            user_name=user_name,
            body_bytes=body_bytes,
        )
    )

    payload = client.request_json("POST", f"{crm_base_url}/hmac-login", headers=headers, body=body)
    sid = _extract_sid(payload)
    if not sid:
        raise ValidationError(f"Could not extract sid from hmac-login response: {json.dumps(payload)[:500]}")
    return sid


def _crm_list_records(
    client: JsonHttpClient,
    *,
    module: str,
    crm_base_url: str,
    sid: str,
    page_size: int,
    max_records: int | None,
    hmac_key_id: str,
    hmac_secret: str,
    user_id: str,
    user_name: str,
) -> dict[str, RecordRow]:
    records: dict[str, RecordRow] = {}
    offset = 0

    while True:
        if max_records is not None and len(records) >= max_records:
            break

        current_limit = page_size
        if max_records is not None:
            current_limit = min(current_limit, max_records - len(records))
            if current_limit <= 0:
                break

        body = {
            "limit": current_limit,
            "offset": offset,
            "columns": None,
            "order": [],
            "saved_search_id": "",
            "prefix": None,
            "listview_type": "list",
            "viewType": "desktop",
            "filter": {"operator": "and", "operands": []},
            "include_hash": True,
            "hash_algorithm": "sha256",
        }

        headers = {
            "Accept": "application/json",
            "sid": sid,
        }
        if hmac_secret and user_id:
            headers.update(
                _build_hmac_headers(
                    key_id=hmac_key_id,
                    secret=hmac_secret,
                    user_id=user_id,
                    user_name=user_name,
                    body_bytes=_json_dumps(body),
                )
            )

        payload = client.request_json("POST", f"{crm_base_url}/list/{module}", headers=headers, body=body)
        rows = payload.get("records")
        if not isinstance(rows, list):
            raise ValidationError(f"Unexpected CRM list shape for {module} at offset={offset}: missing 'records' list")

        if not rows:
            break

        for row in rows:
            if not isinstance(row, dict):
                continue
            record_id = str(row.get("id") or "").strip()
            if not record_id:
                continue
            records[record_id] = RecordRow(
                record_id=record_id,
                name=_to_record_name(row),
                date_modified=str(row.get("date_modified") or "").strip(),
            )

        print(f"CRM {module} page offset={offset} fetched={len(rows)} accumulated={len(records)}")

        if len(rows) < current_limit:
            break
        offset += len(rows)

    return records


def _qdrant_records(
    client: JsonHttpClient,
    *,
    module: str,
    qdrant_url: str,
    qdrant_api_key: str,
    collection: str,
    page_size: int,
    max_records: int | None,
) -> dict[str, RecordRow]:
    records: dict[str, RecordRow] = {}
    offset: Any = None
    target_module = module.lower()

    while True:
        body: dict[str, Any] = {
            "limit": page_size,
            "with_payload": True,
            "with_vector": False,
        }
        if offset is not None:
            body["offset"] = offset

        qdrant_headers = {"Accept": "application/json"}
        if qdrant_api_key:
            qdrant_headers["api-key"] = qdrant_api_key
            qdrant_headers["Authorization"] = f"Bearer {qdrant_api_key}"

        payload = client.request_json(
            "POST",
            f"{qdrant_url}/collections/{parse.quote(collection, safe='')}/points/scroll",
            headers=qdrant_headers,
            body=body,
        )
        result = payload.get("result")
        if not isinstance(result, dict):
            raise ValidationError("Unexpected Qdrant response: missing result object")

        points = result.get("points")
        if not isinstance(points, list):
            raise ValidationError("Unexpected Qdrant response: missing result.points list")

        for point in points:
            if not isinstance(point, dict):
                continue
            q_payload = point.get("payload")
            if not isinstance(q_payload, dict):
                continue

            q_module = str(q_payload.get("module") or "").strip().lower()
            if q_module != target_module:
                continue

            record_id = str(q_payload.get("record_id") or "").strip()
            record_data = q_payload.get("record")
            if (not record_id) and isinstance(record_data, dict):
                record_id = str(record_data.get("id") or "").strip()
            if not record_id:
                continue

            if isinstance(record_data, dict):
                name = _to_record_name(record_data)
                date_modified = str(record_data.get("date_modified") or q_payload.get("modified_at") or "").strip()
            else:
                name = "<unknown>"
                date_modified = str(q_payload.get("modified_at") or "").strip()

            records[record_id] = RecordRow(
                record_id=record_id,
                name=name,
                date_modified=date_modified,
            )

        print(f"Qdrant page fetched={len(points)} accumulated_{module}={len(records)}")

        if max_records is not None and len(records) >= max_records:
            break

        offset = result.get("next_page_offset")
        if offset is None:
            break

    return records


def _print_samples(*, title: str, records: list[RecordRow], max_items: int) -> None:
    print(title)
    if not records:
        print("  - none")
        return
    for row in records[:max_items]:
        date_suffix = f" | modified={row.date_modified}" if row.date_modified else ""
        print(f"  - {row.record_id} | {row.name}{date_suffix}")
    remaining = len(records) - max_items
    if remaining > 0:
        print(f"  ... and {remaining} more")


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate CRM Records vs Qdrant Records ingest completeness")
    parser.add_argument("--module", default="Contacts", help="CRM Module name to check (e.g. Contacts, Meetings, Accounts)")
    parser.add_argument("--tenant-id", default="ai-local", help="Tenant id used for default collection naming")
    parser.add_argument("--collection", default=None, help="Explicit Qdrant collection name")
    parser.add_argument("--collection-prefix", default=None, help="Qdrant collection prefix used with --tenant-id")
    parser.add_argument("--qdrant-url", default=None, help="Qdrant base URL")
    parser.add_argument("--qdrant-api-key", default=None, help="Qdrant API key (or set QDRANT_API_KEY in env)")

    parser.add_argument("--crm-path", default=None, help="Path to rest_coripo (used to read .env defaults)")
    parser.add_argument("--crm-base-url", default=None, help="CRM public API base URL, e.g. http://localhost:2000/public")
    parser.add_argument("--sid", default=None, help="Existing Coripo SID (skip hmac-login)")
    parser.add_argument("--hmac-key-id", default=None, help="Coripo HMAC key id")
    parser.add_argument("--hmac-secret", default=None, help="Coripo HMAC secret")
    parser.add_argument("--user-id", default=None, help="Coripo user id for HMAC")
    parser.add_argument("--user-name", default=None, help="Coripo user name for HMAC")

    parser.add_argument("--page-size", type=int, default=500, help="Page size for CRM and Qdrant scrolling")
    parser.add_argument("--max-records", type=int, default=None, help="Optional cap for debugging")
    parser.add_argument("--sample", type=int, default=20, help="How many diff rows to print")
    parser.add_argument("--check-id", action="append", default=[], help="Specific record id to verify (can repeat)")
    parser.add_argument("--output-json", default=None, help="Write full diff report to JSON file")
    parser.add_argument("--fail-on-missing", action="store_true", help="Exit non-zero when CRM records are missing in Qdrant")

    args = parser.parse_args()

    if args.page_size <= 0:
        raise SystemExit("--page-size must be > 0")

    defaults = _build_runtime_defaults(args)

    client = JsonHttpClient(timeout=90.0)
    sid = (args.sid or "").strip()
    if not sid:
        sid = _login_coripo(
            client,
            crm_base_url=defaults["crm_base_url"],
            hmac_key_id=defaults["hmac_key_id"],
            hmac_secret=defaults["hmac_secret"],
            user_id=defaults["user_id"],
            user_name=defaults["user_name"],
        )
        print("Authenticated via /hmac-login")
    else:
        print("Using provided SID")

    print(f"Module: {args.module}")
    print(f"CRM base URL: {defaults['crm_base_url']}")
    print(f"Qdrant URL: {defaults['qdrant_url']}")
    print(f"Qdrant auth: {'api-key' if defaults['qdrant_api_key'] else 'none'}")
    print(f"Qdrant collection: {defaults['collection']}")

    crm_records = _crm_list_records(
        client,
        module=args.module,
        crm_base_url=defaults["crm_base_url"],
        sid=sid,
        page_size=args.page_size,
        max_records=args.max_records,
        hmac_key_id=defaults["hmac_key_id"],
        hmac_secret=defaults["hmac_secret"],
        user_id=defaults["user_id"],
        user_name=defaults["user_name"],
    )

    qdrant_records = _qdrant_records(
        client,
        module=args.module,
        qdrant_url=defaults["qdrant_url"],
        qdrant_api_key=defaults["qdrant_api_key"],
        collection=defaults["collection"],
        page_size=args.page_size,
        max_records=args.max_records,
    )

    crm_ids = set(crm_records.keys())
    qdrant_ids = set(qdrant_records.keys())

    missing_ids = sorted(crm_ids - qdrant_ids)
    stale_ids = sorted(qdrant_ids - crm_ids)

    missing_rows = [crm_records[item] for item in missing_ids]
    stale_rows = [qdrant_records[item] for item in stale_ids]

    print("\nSummary")
    print(f"  CRM {args.module}: {len(crm_ids)}")
    print(f"  Qdrant {args.module}: {len(qdrant_ids)}")
    print(f"  Missing in Qdrant (CRM-only): {len(missing_ids)}")
    print(f"  Stale in Qdrant (Qdrant-only): {len(stale_ids)}")

    # Only include the default contact ID if module is Contacts
    checks = []
    if args.module.lower() == "contacts":
        checks.append("105004f2-f220-2bbb-2ca0-64e330763f18")
    checks.extend(args.check_id)
    checks = list(dict.fromkeys(checks))

    if checks:
        print("\nRecord checks")
        for record_id in checks:
            in_crm = record_id in crm_ids
            in_qdrant = record_id in qdrant_ids
            status = "OK" if (in_crm and in_qdrant) else "MISSING_IN_QDRANT" if in_crm else "NOT_IN_CRM"
            print(f"  {record_id}: crm={in_crm} qdrant={in_qdrant} status={status}")

    _print_samples(title=f"\nSample missing records (CRM-only):", records=missing_rows, max_items=max(args.sample, 0))
    _print_samples(title=f"\nSample stale records (Qdrant-only):", records=stale_rows, max_items=max(args.sample, 0))

    if missing_ids:
        print("\nSuggested backfill")
        print("  curl -X POST 'http://127.0.0.1:8011/rag/ingest/' \\")
        print("    -H 'Content-Type: application/json' \\")
        print(f"    -H 'X-Tenant: {args.tenant_id}' \\")
        print(f"    -H 'X-User-Id: {defaults['user_id'] or '<user-id>'}' \\")
        if defaults["user_name"]:
            print(f"    -H 'X-User-Name: {defaults['user_name']}' \\")
        print(f"    --data '{{\"modules\":[\"{args.module}\"],\"incremental\":false,\"synchronous\":true,\"page_size\":500}}'")

    if args.output_json:
        output_path = Path(args.output_json).expanduser().resolve()
        report = {
            "module": args.module,
            "crm_base_url": defaults["crm_base_url"],
            "qdrant_url": defaults["qdrant_url"],
            "collection": defaults["collection"],
            "counts": {
                f"crm_{args.module.lower()}": len(crm_ids),
                f"qdrant_{args.module.lower()}": len(qdrant_ids),
                "missing_in_qdrant": len(missing_ids),
                "stale_in_qdrant": len(stale_ids),
            },
            "record_checks": [
                {
                    "record_id": record_id,
                    "in_crm": record_id in crm_ids,
                    "in_qdrant": record_id in qdrant_ids,
                }
                for record_id in checks
            ],
            "missing_in_qdrant": [row.__dict__ for row in missing_rows],
            "stale_in_qdrant": [row.__dict__ for row in stale_rows],
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nWrote report: {output_path}")

    if args.fail_on_missing and missing_ids:
        return 2
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValidationError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
