from __future__ import annotations

import asyncio
import base64
import copy
import hashlib
import hmac
import json
import re
import time
import uuid
from datetime import datetime, timezone
from typing import Any

import httpx

from app.core.config import get_settings
from app.core.http import get_shared_http_client
from app.core.logging import get_logger
from app.utils.modules import canonical_module_name


logger = get_logger(__name__)

_UUID_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)

# HMAC-derived SIDs cached across CoripoClient instances so each API request
# does not pay an extra hmac-login round trip. Keyed by (api_base, key_id, user_id).
_SID_CACHE: dict[tuple[str, str, str], tuple[float, str]] = {}


class CoripoClient:
    """Coripo public API client used by the app."""

    def __init__(
        self,
        base_url: str,
        token: str,
        timeout: float | None = None,
        user_id: str | None = None,
        user_name: str | None = None,
    ):
        settings = get_settings()
        self.settings = settings
        self.base_url = (base_url or "").strip().rstrip("/")
        self.token = (token or "").strip()
        self.timeout = timeout if timeout is not None else settings.crm_timeout_seconds

        self.mode = "coripo_public"
        self.is_v4_1 = False

        self._auth_config = self._parse_auth_config(self.token)
        self.user_id = str(user_id or self._auth_config.get("user_id") or "").strip() or None
        self.user_name = str(user_name or self._auth_config.get("user_name") or "").strip() or None

        self.coripo_hmac_key_id = str(
            self._auth_config.get("hmac_key_id") or settings.coripo_hmac_key_id
        ).strip()
        self.coripo_hmac_secret = str(self._auth_config.get("hmac_secret") or "").strip()
        if not self.coripo_hmac_secret:
            fallback_secret = str(settings.hmac_keys.get(self.coripo_hmac_key_id) or "").strip()
            if fallback_secret:
                self.coripo_hmac_secret = fallback_secret
                logger.warning(
                    "CoripoClient using HMAC secret fallback from HMAC_KEYS_JSON for key_id=%s",
                    self.coripo_hmac_key_id,
                )
        self.clean_response_default = bool(self._auth_config.get("clean_response", True))
        self._coripo_session_id = str(
            self._auth_config.get("session_id")
            or self._auth_config.get("sid")
            or ""
        ).strip() or None

        if self._is_uuid(self.token):
            self._coripo_session_id = self.token
        elif ":" in self.token and not self.token.startswith("{"):
            # "keyId:secret" compact token format.
            kid, secret = self.token.split(":", 1)
            if kid.strip() and secret.strip():
                self.coripo_hmac_key_id = kid.strip()
                self.coripo_hmac_secret = secret.strip()
        elif self.token and not self.token.startswith("{") and not self.coripo_hmac_secret:
            # Plain token means HMAC secret for convenience.
            self.coripo_hmac_secret = self.token

        # Prefer a fresh HMAC-derived SID over any pre-configured static one.
        # Static SIDs may have limited scope or become stale after a server restart;
        # HMAC login always yields a fresh, full-scope session.
        if self._coripo_session_id and self._can_use_hmac():
            self._coripo_session_id = None

        self._sid_from_cache = False
        self._sid_lock = asyncio.Lock()

    @staticmethod
    def _is_uuid(value: str | None) -> bool:
        return bool(value and _UUID_RE.match(value.strip()))

    @staticmethod
    def _parse_auth_config(token: str) -> dict[str, Any]:
        if not token:
            return {}
        if token.startswith("{"):
            try:
                parsed = json.loads(token)
                if isinstance(parsed, dict):
                    # Compatibility: allow {"keyId":"secret"} mapping shape.
                    if "hmac_secret" not in parsed and "session_id" not in parsed and "sid" not in parsed:
                        if len(parsed) == 1:
                            only_key = next(iter(parsed.keys()))
                            only_val = parsed.get(only_key)
                            if str(only_key).strip() and str(only_val or "").strip():
                                return {
                                    "mode": "coripo_public",
                                    "hmac_key_id": str(only_key).strip(),
                                    "hmac_secret": str(only_val).strip(),
                                }
                    return parsed
            except json.JSONDecodeError:
                return {}
        if ":" in token:
            key_id, secret = token.split(":", 1)
            if key_id.strip() and secret.strip():
                return {
                    "mode": "coripo_public",
                    "hmac_key_id": key_id.strip(),
                    "hmac_secret": secret.strip(),
                }
        return {}

    def _coripo_api_base(self) -> str:
        base = self.base_url.rstrip("/")
        if not base:
            return "http://localhost:2000/public"
        if base.endswith("/public/index.php"):
            return base.removesuffix("/index.php")
        if base.endswith("/public"):
            return base
        if "/public/" in base:
            return base.split("/public/", 1)[0] + "/public"
        return f"{base}/public"

    def _can_use_hmac(self) -> bool:
        return bool(self.coripo_hmac_secret and self.user_id)

    def _build_coripo_hmac_headers(self, body_bytes: bytes) -> dict[str, str]:
        if not self._can_use_hmac():
            return {}

        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        nonce = str(uuid.uuid4())
        sign_base = f"{timestamp}|{nonce}|{self.user_id}|".encode("utf-8") + body_bytes
        signature = base64.b64encode(
            hmac.new(
                self.coripo_hmac_secret.encode("utf-8"),
                sign_base,
                hashlib.sha256,
            ).digest()
        ).decode("utf-8")

        headers = {
            "Authorization": f"HMAC keyId={self.coripo_hmac_key_id}, signature={signature}",
            "X-Timestamp": timestamp,
            "X-Nonce": nonce,
            "X-User-Id": str(self.user_id),
        }
        if self.user_name:
            headers["X-User-Name"] = str(self.user_name)
        return headers

    @staticmethod
    def _decode_response(response: httpx.Response) -> dict[str, Any]:
        if not response.content:
            return {}
        content_type = response.headers.get("content-type", "")
        if "json" not in content_type.lower():
            return {"raw": response.text}
        parsed = response.json()
        if isinstance(parsed, dict):
            return parsed
        if isinstance(parsed, list):
            return {"records": parsed}
        return {"data": parsed}

    @staticmethod
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
            inner = message.get("data")
            if isinstance(inner, dict):
                for key in ("sid", "session_id"):
                    value = inner.get(key)
                    if isinstance(value, str) and value.strip():
                        return value.strip()

        data = payload.get("data")
        if isinstance(data, dict):
            for key in ("sid", "session_id"):
                value = data.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
        return None

    def _sid_cache_key(self) -> tuple[str, str, str]:
        return (self._coripo_api_base(), self.coripo_hmac_key_id, str(self.user_id or ""))

    def _invalidate_cached_sid(self) -> None:
        _SID_CACHE.pop(self._sid_cache_key(), None)
        self._coripo_session_id = None
        self._sid_from_cache = False

    async def _ensure_coripo_sid(self) -> None:
        if self._coripo_session_id:
            return
        if not self._can_use_hmac():
            raise ValueError(
                "Coripo authentication requires either SID token or HMAC secret + user_id."
            )

        # The lock keeps concurrent calls on this instance (e.g. gathered module
        # searches) from racing into parallel hmac-login round trips.
        async with self._sid_lock:
            if self._coripo_session_id:
                return

            ttl = self.settings.coripo_sid_cache_ttl_seconds
            cache_key = self._sid_cache_key()
            if ttl > 0:
                cached = _SID_CACHE.get(cache_key)
                if cached and cached[0] > time.monotonic():
                    self._coripo_session_id = cached[1]
                    self._sid_from_cache = True
                    return

            sid = await self._login_for_sid()
            self._coripo_session_id = sid
            self._sid_from_cache = False
            if ttl > 0:
                _SID_CACHE[cache_key] = (time.monotonic() + ttl, sid)

    async def _login_for_sid(self) -> str:
        # Coripo HMAC signing is computed from exact raw body bytes.
        # Preferred flow (per rest_coripo/test_hmac_public_login.sh) is hmac-login.
        attempts: list[tuple[str, dict[str, Any] | None]] = [
            ("POST", {}),   # /hmac-login with "{}"
            ("GET", None),  # /hmac-login as GET
        ]
        last_exc: Exception | None = None
        for method, body in attempts:
            try:
                response = await self._coripo_request(
                    method,
                    "hmac-login",
                    json_body=body,
                    require_sid=False,
                )
                sid = self._extract_sid(response)
                if sid:
                    return sid
            except httpx.HTTPStatusError as exc:
                last_exc = exc
                continue

        # Backward-compatible fallback for older deployments.
        fallback_attempts: list[tuple[str, dict[str, Any] | None]] = [
            ("POST", {}),
            ("POST", None),
            ("GET", None),
        ]
        for method, body in fallback_attempts:
            try:
                response = await self._coripo_request(
                    method,
                    "checksid",
                    json_body=body,
                    require_sid=False,
                )
                sid = self._extract_sid(response)
                if sid:
                    return sid
            except httpx.HTTPStatusError as exc:
                last_exc = exc
                continue

        if last_exc is not None:
            raise last_exc
        raise ValueError("Coripo hmac-login/checksid did not return sid")

    async def _coripo_request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
        require_sid: bool = True,
    ) -> dict[str, Any]:
        if require_sid and not self._coripo_session_id:
            await self._ensure_coripo_sid()

        body_bytes = b""
        if json_body is not None:
            body_bytes = json.dumps(
                json_body,
                ensure_ascii=False,
                separators=(",", ":"),
                default=str,
            ).encode("utf-8")

        url = f"{self._coripo_api_base()}/{path.lstrip('/')}"

        async def _send() -> httpx.Response:
            headers: dict[str, str] = {"Accept": "application/json"}
            if self._coripo_session_id:
                headers["sid"] = self._coripo_session_id
            if json_body is not None:
                headers["Content-Type"] = "application/json"
            headers.update(self._build_coripo_hmac_headers(body_bytes))
            return await get_shared_http_client().request(
                method=method,
                url=url,
                headers=headers,
                params=params,
                content=body_bytes if json_body is not None else None,
                timeout=self.timeout,
            )

        response = await _send()

        if (
            response.status_code in {401, 403}
            and require_sid
            and self._sid_from_cache
            and self._can_use_hmac()
        ):
            # A SID taken from the cross-request cache may have expired server-side.
            # Drop it and retry once with a freshly negotiated session.
            self._invalidate_cached_sid()
            await self._ensure_coripo_sid()
            response = await _send()

        if response.status_code >= 400:
            logger.error(
                "Coripo request failed status=%s method=%s url=%s body=%s",
                response.status_code,
                method,
                url,
                response.text,
            )
            response.raise_for_status()

        return self._decode_response(response)

    @staticmethod
    def _canonical_module(module: str) -> str:
        value = str(module or "").strip().strip("/")
        if not value:
            raise ValueError("Module cannot be empty")
        return canonical_module_name(value)

    @staticmethod
    def _module_candidates(module_name: str) -> list[str]:
        value = str(module_name or "").strip().strip("/")
        if not value:
            return []

        candidates = [value]
        if "_" in value:
            parts = value.split("_")
            title_parts = [part[:1].upper() + part[1:] if part else part for part in parts]
            candidates.append("_".join(title_parts))
            if parts and parts[0] and parts[0].isalpha() and len(parts[0]) <= 4:
                prefixed = [parts[0].upper(), *title_parts[1:]]
                candidates.append("_".join(prefixed))
        candidates.append(value[:1].upper() + value[1:])
        candidates.append(value.upper())
        return list(dict.fromkeys(candidates))

    @staticmethod
    def _default_cont_filter(query: str) -> dict[str, Any]:
        return {
            "operator": "and",
            "operands": [
                {
                    "operator": "and",
                    "operands": [
                        {
                            "field": "*",
                            "fieldModule": None,
                            "fieldRel": None,
                            "type": "cont",
                            "value": query,
                            "relationField": None,
                        }
                    ],
                }
            ],
        }

    @staticmethod
    def _parse_order_by(order_by: Any, module_name: str) -> list[dict[str, str]]:
        text = str(order_by or "").strip()
        if not text:
            return []
        # Supports common style: "accounts.date_modified DESC"
        parts = text.split()
        field_part = parts[0]
        sort = "ASC"
        if len(parts) > 1 and parts[1].upper() in {"ASC", "DESC"}:
            sort = parts[1].upper()
        if "." in field_part:
            _, field_name = field_part.split(".", 1)
        else:
            field_name = field_part
        field_name = field_name.strip()
        if not field_name:
            return []
        return [{"field": field_name, "sort": sort, "module": module_name}]

    def _build_coripo_list_payload(self, module_name: str, data: dict[str, Any]) -> dict[str, Any]:
        source = copy.deepcopy(data or {})

        limit = int(source.pop("max_results", source.get("limit", 20)) or 20)
        offset = int(source.get("offset", 0) or 0)
        limit = max(1, min(limit, 500))

        query = str(source.get("q") or source.get("query") or "").strip()
        order = source.get("order")
        if not isinstance(order, list):
            order = self._parse_order_by(source.get("order_by"), module_name)

        payload: dict[str, Any] = {
            "limit": limit,
            "offset": offset,
            "columns": source.get("columns", None),
            # "columns": None, # None means default list view columns; can be overridden by "columns" key in source
            "order": order,
            # "groupBy": source.get("groupBy", []),
            # "function": source.get("function", {}),
            # "alterName": source.get("alterName", {}),
            # "groupByDate": source.get("groupByDate", []),
            # "savedSearch": bool(source.get("savedSearch", True)),
            "saved_search_id": "", # should be empty do display default list view
            "prefix": None,
            "listview_type": "list",
            "viewType": "desktop",
        }

        if "recordsCount" in source:
            payload["recordsCount"] = bool(source["recordsCount"])
        if "listview_type" in source:
            payload["listview_type"] = source["listview_type"]

        if isinstance(source.get("filter"), dict):
            payload["filter"] = source["filter"]
        elif query:
            payload["filter"] = self._default_cont_filter(query)
        else:
            payload["filter"] = {"operator": "and", "operands": []}

        if "include_hash" in source:
            payload["include_hash"] = bool(source.get("include_hash"))
        if source.get("hash_algorithm"):
            payload["hash_algorithm"] = str(source["hash_algorithm"])

        return payload

    @staticmethod
    def _extract_records(value: Any) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        record_keys = {"records", "entry_list", "items", "results"}

        def walk(node: Any, parent_key: str = "") -> None:
            if isinstance(node, list):
                if parent_key in record_keys:
                    for item in node:
                        if isinstance(item, dict):
                            records.append(item)
                for item in node:
                    walk(item, parent_key=parent_key)
                return

            if not isinstance(node, dict):
                return

            for key, child in node.items():
                walk(child, parent_key=key)

        walk(value, parent_key="")

        unique: dict[str, dict[str, Any]] = {}
        passthrough: list[dict[str, Any]] = []
        for row in records:
            row_id = str(row.get("id") or row.get("record_id") or "").strip()
            if row_id:
                if row_id not in unique:
                    unique[row_id] = row
            else:
                passthrough.append(row)
        return list(unique.values()) + passthrough

    @staticmethod
    def _clean_record(
        record: dict[str, Any],
        *,
        include_fields: set[str] | None = None,
    ) -> dict[str, Any]:
        cleaned: dict[str, Any] = {}
        for key, value in record.items():
            field = str(key).strip()
            if not field:
                continue
            if include_fields is not None and field not in include_fields:
                continue

            # Drop bridge/noise keys like "accounts|id", "email_addresses_primary|id".
            if "|" in field:
                continue

            # Keep deterministic change hash for incremental ingest.
            if field.startswith("_") and field != "_record_hash":
                continue

            # Drop empty values by default to reduce LLM context noise.
            if value is None:
                continue
            if isinstance(value, str):
                stripped = value.strip()
                if not stripped:
                    continue
                cleaned[field] = stripped
                continue
            if isinstance(value, (list, dict)) and not value:
                continue

            # Keep only scalar/list payloads; nested objects are usually metadata noise.
            if isinstance(value, dict):
                continue
            cleaned[field] = value

        # Always preserve primary ID when present, even if empty filtering removed it.
        if "id" in record and isinstance(record.get("id"), str) and record["id"].strip():
            cleaned["id"] = record["id"].strip()
        return cleaned

    def _clean_records(
        self,
        records: list[dict[str, Any]],
        *,
        include_fields: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        cleaned: list[dict[str, Any]] = []
        for row in records:
            if not isinstance(row, dict):
                continue
            sanitized = self._clean_record(row, include_fields=include_fields)
            if sanitized:
                cleaned.append(sanitized)
        return cleaned

    @staticmethod
    def _extract_first_record(payload: dict[str, Any]) -> dict[str, Any] | None:
        records = CoripoClient._extract_records(payload)
        if records:
            return records[0]

        for key in ("record", "item"):
            value = payload.get(key)
            if isinstance(value, dict):
                return value
        data = payload.get("data")
        if isinstance(data, dict):
            for key in ("record", "item"):
                value = data.get(key)
                if isinstance(value, dict):
                    return value
        return None

    @staticmethod
    def _extract_total(payload: dict[str, Any]) -> int | None:
        keys = ("total", "total_count", "count", "recordsCount")
        stack = [payload]
        while stack:
            node = stack.pop()
            if not isinstance(node, dict):
                continue
            for key in keys:
                value = node.get(key)
                if isinstance(value, bool):
                    continue
                if isinstance(value, int):
                    return value
                if isinstance(value, str) and value.isdigit():
                    return int(value)
            for nested_key in ("data", "message", "result"):
                nested = node.get(nested_key)
                if isinstance(nested, dict):
                    stack.append(nested)
        return None

    @staticmethod
    def _extract_record_id(payload: dict[str, Any]) -> str | None:
        keys = ("id", "record_id")
        stack = [payload]
        while stack:
            node = stack.pop()
            if not isinstance(node, dict):
                continue
            for key in keys:
                value = node.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
            for nested_key in ("data", "message", "result", "record"):
                nested = node.get(nested_key)
                if isinstance(nested, dict):
                    stack.append(nested)
        return None

    @staticmethod
    def _sanitize_coripo_fields(data: dict[str, Any]) -> dict[str, Any]:
        if isinstance(data.get("fields"), dict):
            source = dict(data["fields"])
        else:
            source = dict(data)
        reserved = {
            "id",
            "fields",
            "relationships",
            "customData",
            "files",
            "invitees",
            "inviteesBackup",
            "requested_by_user_id",
            "confirm_action",
            "contact_name",
            "related_contact_name",
            "invite_contact_name",
            "participant_name",
            "account_name",
            "related_account_name",
            "company_name",
            "company",
        }
        return {k: v for k, v in source.items() if k not in reserved and v is not None}

    def _build_set_payload(self, data: dict[str, Any], *, include_invitees_backup: bool) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "fields": self._sanitize_coripo_fields(data),
            "relationships": data.get("relationships", {}),
            "customData": data.get("customData", {}),
            "files": data.get("files", []),
            "invitees": data.get("invitees", {"Users": [], "Contacts": [], "Leads": []}),
        }
        if include_invitees_backup:
            payload["inviteesBackup"] = data.get(
                "inviteesBackup",
                data.get("invitees", {"Users": [], "Contacts": [], "Leads": []}),
            )
        return payload

    def _sanitize_list_response(
        self,
        module_name: str,
        payload: dict[str, Any],
        raw: dict[str, Any],
        *,
        clean_records: bool,
        include_fields: set[str] | None = None,
        include_field_names: bool = True,
    ) -> dict[str, Any]:
        source_records = raw.get("records")
        if isinstance(source_records, list):
            # Prefer canonical Coripo list payload shape when present. The generic
            # recursive extractor can collapse rows in mixed payloads, which then
            # breaks offset-based pagination for ingest callers.
            records = [row for row in source_records if isinstance(row, dict)]
            source_record_count = len(source_records)
        else:
            records = self._extract_records(raw)
            source_record_count = len(records)
        if clean_records:
            records = self._clean_records(records, include_fields=include_fields)

        output: dict[str, Any] = {
            "records": records,
            "source_record_count": source_record_count,
        }
        if include_field_names:
            output["column_fields"] = self._extract_column_field_names(payload, raw)
            output["def_fields"] = self._extract_def_field_names(raw)
        return output

    @staticmethod
    def _extract_column_field_names(payload: dict[str, Any], raw: dict[str, Any]) -> list[str]:
        names: list[str] = []

        payload_columns = payload.get("columns")
        if isinstance(payload_columns, list):
            for item in payload_columns:
                if isinstance(item, dict):
                    field = str(item.get("field") or "").strip()
                    if field:
                        names.append(field)

        if not names:
            rows = raw.get("rows")
            if isinstance(rows, dict):
                for key in rows.keys():
                    field = str(key or "").strip()
                    if field:
                        names.append(field.lower())

        # Stable compact unique list.
        return list(dict.fromkeys(names))

    @staticmethod
    def _extract_def_field_names(raw: dict[str, Any]) -> list[str]:
        defs = raw.get("def")
        if not isinstance(defs, dict):
            return []
        names = [str(key).strip() for key in defs.keys() if str(key).strip()]
        return list(dict.fromkeys(names))

    @staticmethod
    def _extract_virtual_ids(data: dict[str, Any]) -> list[str] | None:
        raw_ids = data.get("ids")
        if raw_ids is None:
            return None
        if not isinstance(raw_ids, list):
            raise ValueError("Virtual id batch lookup requires ids to be an array.")

        ids = [str(item or "").strip() for item in raw_ids]
        ids = [item for item in ids if item]
        if not ids:
            raise ValueError("Virtual id batch lookup requires at least one id.")
        return list(dict.fromkeys(ids))

    @staticmethod
    def _sort_value(value: Any) -> tuple[int, str]:
        if value is None:
            return (1, "")
        if isinstance(value, str):
            return (0, value.lower())
        return (0, str(value).lower())

    def _apply_local_order(self, records: list[dict[str, Any]], order: Any) -> list[dict[str, Any]]:
        if not isinstance(order, list) or not order:
            return records

        ordered = list(records)
        for spec in reversed(order):
            if not isinstance(spec, dict):
                continue
            field = str(spec.get("field") or "").strip()
            if not field:
                continue
            reverse = str(spec.get("sort") or "ASC").strip().upper() == "DESC"
            ordered.sort(key=lambda row: self._sort_value(row.get(field)), reverse=reverse)
        return ordered

    async def _fetch_records_by_ids(self, module_name: str, ids: list[str]) -> list[dict[str, Any]]:
        async def _fetch_one(record_id: str) -> dict[str, Any] | None:
            try:
                raw = await self._coripo_request(
                    "GET",
                    f"detail/{module_name}/{record_id}",
                    require_sid=True,
                )
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code == 404:
                    logger.info("Coripo detail lookup skipped missing %s/%s", module_name, record_id)
                    return None
                raise

            record = self._extract_first_record(raw)
            if record is None:
                return {"id": record_id}
            if isinstance(record, dict) and "id" not in record:
                record = dict(record)
                record["id"] = record_id
            return record if isinstance(record, dict) else {"id": record_id}

        results = await asyncio.gather(*(_fetch_one(record_id) for record_id in ids))
        return [row for row in results if isinstance(row, dict)]

    def _sanitize_mutation_response(
        self,
        raw: dict[str, Any],
        *,
        fallback_id: str | None = None,
        clean_record: bool = True,
        include_fields: set[str] | None = None,
    ) -> dict[str, Any]:
        response: dict[str, Any] = {"status": "ok"}
        record_id = self._extract_record_id(raw) or (fallback_id.strip() if fallback_id else None)
        if record_id:
            response["id"] = record_id
        record = self._extract_first_record(raw)
        if record is not None:
            response["record"] = (
                self._clean_record(record, include_fields=include_fields) if clean_record else record
            )
        return response

    @staticmethod
    def _resolve_company_overview_currency(payload: dict[str, Any]) -> str:
        direct_currency = str(payload.get("monetary_values_currency") or "").strip().upper()
        if direct_currency:
            return direct_currency

        default_currency = payload.get("default_currency")
        if isinstance(default_currency, dict):
            for key in ("iso4217", "code"):
                value = str(default_currency.get(key) or "").strip().upper()
                if value:
                    return value

        # Coripo/Sugar defaults monetary values to CRM default currency. Keep a
        # deterministic fallback when explicit currency metadata is missing.
        return "CZK"

    @classmethod
    def _annotate_amount_fields(cls, node: Any, currency: str) -> None:
        if isinstance(node, list):
            for item in node:
                cls._annotate_amount_fields(item, currency)
            return

        if not isinstance(node, dict):
            return

        raw_amount = node.get("amount_usdollar")
        if raw_amount is not None and str(raw_amount).strip():
            if node.get("amount") is None or str(node.get("amount")).strip() == "":
                node["amount"] = raw_amount
            resolved_currency = str(node.get("amount_currency") or currency).strip().upper() or currency
            node["amount_currency"] = resolved_currency
            if not str(node.get("amount_display") or "").strip():
                node["amount_display"] = f"{node['amount']} {resolved_currency}"
            node.setdefault("amount_source_field", "amount_usdollar")

        for value in node.values():
            cls._annotate_amount_fields(value, currency)

    @classmethod
    def _normalize_company_overview_payload(cls, payload: dict[str, Any]) -> dict[str, Any]:
        currency = cls._resolve_company_overview_currency(payload)
        payload["monetary_values_currency"] = currency
        cls._annotate_amount_fields(payload, currency)
        return payload

    async def generic_search(self, query: str, scope: str = "all") -> dict[str, Any]:
        module_map = {
            "contacts": "Contacts",
            "accounts": "Accounts",
            "meetings": "Meetings",
            "opportunities": "Opportunities",
            "opportunites": "Opportunities",
            "quotes": "Quotes",
            "acm_invoices": "acm_invoices",
        }
        normalized_scope = str(scope or "all").strip().lower()
        payload_data = {"query": query, "q": query, "limit": 20, "offset": 0}

        if normalized_scope in module_map:
            return await self.execute_module_action(
                module=module_map[normalized_scope],
                action="list",
                data=payload_data,
            )

        results = await asyncio.gather(
            *(
                self.execute_module_action(module=module_name, action="list", data=payload_data)
                for module_name in module_map.values()
            )
        )
        return dict(zip(module_map.keys(), results))

    async def execute_module_action(
        self,
        module: str,
        action: str,
        data: dict[str, Any],
    ) -> dict[str, Any]:
        normalized_action = str(action or "").strip().lower()
        module_name = self._canonical_module(module)
        clean_records = bool(data.get("clean_records", self.clean_response_default))
        include_field_names = bool(data.get("include_field_names", True))
        raw_value = data.get("response_fields")
        include_fields: set[str] | None = None
        if isinstance(raw_value, list):
            include_fields = {str(item).strip() for item in raw_value if str(item).strip()}
            include_fields.add("id")

        if normalized_action in {"list", "search"}:
            requested_ids = self._extract_virtual_ids(data)
            if requested_ids is not None:
                total_limit = max(1, min(int(data.get("limit") or len(requested_ids)), 500))
                records = await self._fetch_records_by_ids(module_name, requested_ids)
                if clean_records:
                    records = self._clean_records(records, include_fields=include_fields)
                total_count = len(records)
                records = self._apply_local_order(records, data.get("order"))
                records = records[:total_limit]

                output: dict[str, Any] = {
                    "records": records,
                    "source_record_count": total_count,
                }
                if include_field_names:
                    output["column_fields"] = sorted(include_fields) if include_fields else []
                    output["def_fields"] = []
                return output

            payload = self._build_coripo_list_payload(module_name, data)
            last_exc: Exception | None = None
            for candidate in self._module_candidates(module_name):
                try:
                    raw = await self._coripo_request(
                        "POST",
                        f"list/{candidate}",
                        json_body=payload,
                        require_sid=True,
                    )
                    return self._sanitize_list_response(
                        candidate,
                        payload,
                        raw,
                        clean_records=clean_records,
                        include_fields=include_fields,
                        include_field_names=include_field_names,
                    )
                except httpx.HTTPStatusError as exc:
                    last_exc = exc
                    if exc.response.status_code < 500:
                        raise
            if last_exc is not None:
                raise last_exc
            raise ValueError(f"Unable to resolve module for list action: {module_name}")

        if normalized_action == "create":
            payload = self._build_set_payload(data, include_invitees_backup=False)
            raw = await self._coripo_request(
                "POST",
                f"set/{module_name}",
                json_body=payload,
                require_sid=True,
            )
            return self._sanitize_mutation_response(
                raw,
                clean_record=clean_records,
                include_fields=include_fields,
            )

        if normalized_action in {"update", "patch"}:
            record_id = str(data.get("id") or "").strip()
            if not record_id:
                raise ValueError("Update action requires 'id' in data")
            payload = self._build_set_payload(data, include_invitees_backup=True)
            raw = await self._coripo_request(
                "POST",
                f"set/{module_name}/{record_id}",
                json_body=payload,
                require_sid=True,
            )
            return self._sanitize_mutation_response(
                raw,
                fallback_id=record_id,
                clean_record=clean_records,
                include_fields=include_fields,
            )

        if normalized_action == "delete":
            record_id = str(data.get("id") or "").strip()
            if not record_id:
                raise ValueError("Delete action requires 'id' in data")
            await self._coripo_request(
                "DELETE",
                f"delete/{module_name}/{record_id}",
                require_sid=True,
            )
            return {"status": "ok", "id": record_id}

        if normalized_action == "company_overview":
            record_id = str(data.get("id") or "").strip()
            if not record_id:
                raise ValueError("company_overview action requires 'id' in data")
            raw = await self._coripo_request(
                "POST",
                f"detail/{module_name}/{record_id}",
                json_body={"AiRequest": True},
                require_sid=True,
            )
            message = raw.get("message")
            if isinstance(message, dict):
                payload = message.get("data")
                if isinstance(payload, dict):
                    return self._normalize_company_overview_payload(payload)
            data_payload = raw.get("data")
            if isinstance(data_payload, dict):
                return self._normalize_company_overview_payload(data_payload)
            if isinstance(raw, dict):
                return self._normalize_company_overview_payload(raw)
            return raw

        if normalized_action in {"get", "read", "detail"}:
            record_id = str(data.get("id") or "").strip()
            if not record_id:
                raise ValueError("Read action requires 'id' in data")
            raw = await self._coripo_request(
                "GET",
                f"detail/{module_name}/{record_id}",
                require_sid=True,
            )
            record = self._extract_first_record(raw)
            output: dict[str, Any] = {"id": record_id}
            if record is not None:
                output["record"] = (
                    self._clean_record(record, include_fields=include_fields)
                    if clean_records
                    else record
                )
            return output

        # Fallback for uncommon Coripo actions.
        return await self._coripo_request(
            normalized_action.upper(),
            f"{normalized_action}/{module_name}",
            json_body=data if data else None,
            require_sid=True,
        )

    async def get_dynamic_schema(self, module: str) -> dict[str, Any]:
        module_name = self._canonical_module(module)
        raw = await self._coripo_request(
            "GET",
            f"defs/{module_name}",
            require_sid=True,
        )

        defs: dict[str, Any] | None = None
        if isinstance(raw.get("defs"), dict):
            defs = raw["defs"]
        elif isinstance(raw.get("field_defs"), dict):
            defs = raw["field_defs"]
        elif isinstance(raw.get("fields"), dict):
            defs = raw["fields"]
        else:
            data = raw.get("data")
            if isinstance(data, dict):
                if isinstance(data.get("defs"), dict):
                    defs = data["defs"]
                elif isinstance(data.get("field_defs"), dict):
                    defs = data["field_defs"]
                elif isinstance(data.get("fields"), dict):
                    defs = data["fields"]
            message = raw.get("message")
            if defs is None and isinstance(message, dict):
                inner = message.get("data")
                if isinstance(inner, dict):
                    if isinstance(inner.get("defs"), dict):
                        defs = inner["defs"]
                    elif isinstance(inner.get("field_defs"), dict):
                        defs = inner["field_defs"]
                    elif isinstance(inner.get("fields"), dict):
                        defs = inner["fields"]

        return {"module": module_name, "defs": defs or {}}

    async def fetch_ai_ingest_dump(self, module: str) -> dict[str, Any]:
        module_name = self._canonical_module(module)
        offset = 0
        page_size = 500
        all_records: list[dict[str, Any]] = []

        while True:
            payload = await self.execute_module_action(
                module=module_name,
                action="list",
                data={
                    "query": "",
                    "q": "",
                    "offset": offset,
                    "limit": page_size,
                    "include_hash": True,
                    "hash_algorithm": "sha256",
                    "savedSearch": False,
                },
            )
            records = payload.get("records", [])
            if not isinstance(records, list) or not records:
                break

            batch = [row for row in records if isinstance(row, dict)]
            all_records.extend(batch)

            if len(batch) < page_size:
                break
            offset += len(batch)

        return {"records": all_records}
