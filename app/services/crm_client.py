from __future__ import annotations

import base64
import hashlib
import hmac
import json
import re
import uuid
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlparse

import httpx

from app.core.config import get_settings
from app.core.logging import get_logger


logger = get_logger(__name__)

_UUID_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)


class SugarClient:
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
        self.base_url = base_url.rstrip("/")
        self.token = (token or "").strip()
        self.timeout = timeout if timeout is not None else settings.crm_timeout_seconds

        self._auth_config = self._parse_auth_config(self.token)
        self.user_id = user_id or self._auth_config.get("user_id")
        self.user_name = user_name or self._auth_config.get("user_name")
        self._session_id: str | None = None
        self._coripo_session_id: str | None = None

        self.mode = self._detect_mode()
        self.is_v4_1 = self.mode == "sugar_v4_1"

        self.coripo_hmac_key_id = str(
            self._auth_config.get("hmac_key_id") or settings.coripo_hmac_key_id
        )
        self.coripo_hmac_secret = str(self._auth_config.get("hmac_secret") or "")

        if self.mode == "coripo_public":
            # For Coripo we support both HMAC secret and static sid in token.
            if self._is_uuid(self.token):
                self._coripo_session_id = self.token
            elif self.token and not self.token.startswith("{"):
                # Plain token is interpreted as HMAC secret for convenience.
                self.coripo_hmac_secret = self.token

    @staticmethod
    def _is_uuid(value: str | None) -> bool:
        if not value:
            return False
        return _UUID_RE.match(value.strip()) is not None

    @staticmethod
    def _parse_auth_config(token: str) -> dict[str, Any]:
        if not token:
            return {}
        # Support compact "keyId:secret" token format for Coripo HMAC.
        if ":" in token and not token.startswith("{"):
            key_id, secret = token.split(":", 1)
            if key_id.strip() and secret.strip():
                return {
                    "mode": "coripo_public",
                    "hmac_key_id": key_id.strip(),
                    "hmac_secret": secret.strip(),
                }
        if token.startswith("{"):
            try:
                parsed = json.loads(token)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                pass
        return {}

    def _detect_mode(self) -> str:
        explicit_mode = str(self._auth_config.get("mode") or "").strip().lower()
        if explicit_mode in {"coripo", "coripo-public", "coripo_public"}:
            return "coripo_public"
        if explicit_mode in {"v4_1", "sugar-v4_1", "sugar_v4_1", "legacy"}:
            return "sugar_v4_1"
        if explicit_mode in {"modern", "sugar-modern", "sugar_modern"}:
            return "sugar_modern"

        if "/service/v4_1/rest.php" in self.base_url or self.base_url.endswith("/rest.php"):
            return "sugar_v4_1"
        if "/public" in self.base_url:
            return "coripo_public"
        return "sugar_modern"

    @property
    def headers(self) -> dict[str, str]:
        return {
            "OAuth-Token": self.token,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    async def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        url = f"{self.base_url}/{path.lstrip('/')}"
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.request(
                method=method,
                url=url,
                headers=self.headers,
                params=params,
                json=json_body,
            )

        if response.status_code >= 400:
            logger.error(
                "CRM request failed mode=%s status=%s url=%s body=%s",
                self.mode,
                response.status_code,
                url,
                response.text,
            )
            response.raise_for_status()

        return self._decode_response(response)

    @staticmethod
    def _decode_response(response: httpx.Response) -> dict[str, Any]:
        if not response.content:
            return {"ok": True}
        content_type = response.headers.get("content-type", "")
        if "json" not in content_type.lower():
            return {"raw": response.text}
        parsed = response.json()
        return parsed if isinstance(parsed, dict) else {"data": parsed}

    def _canonical_module(self, module: str) -> str:
        normalized = module.strip().strip("/")
        mapping = {
            "contacts": "Contacts",
            "accounts": "Accounts",
            "meetings": "Meetings",
            "calls": "Calls",
            "tasks": "Tasks",
            "notes": "Notes",
            "opportunities": "Opportunities",
            "leads": "Leads",
            "users": "Users",
            "cases": "Cases",
        }
        lowered = normalized.lower()
        if lowered in mapping:
            return mapping[lowered]
        if not normalized:
            return normalized
        return normalized[0].upper() + normalized[1:]

    def _escape_like(self, value: str) -> str:
        return value.replace("\\", "\\\\").replace("'", "\\'")

    # -----------------------------
    # Coripo public REST helpers
    # -----------------------------

    def _coripo_api_base(self) -> str:
        base = self.base_url.rstrip("/")

        if base.endswith("/v1"):
            return base
        if base.endswith("/public"):
            return base
        if base.endswith("/public/index.php"):
            return base.removesuffix("/index.php")

        if "/public/" in base:
            prefix = base.split("/public/", 1)[0] + "/public"
            return prefix

        # Safe default for local deployments where base_url is host root.
        return f"{base}/public"

    def _can_use_coripo_hmac(self) -> bool:
        return (
            self.mode == "coripo_public"
            and bool(self.coripo_hmac_key_id)
            and bool(self.coripo_hmac_secret)
            and bool(self.user_id)
        )

    def _build_coripo_hmac_headers(self, body_bytes: bytes) -> dict[str, str]:
        if not self._can_use_coripo_hmac():
            return {}

        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        nonce = str(uuid.uuid4())
        base = (
            f"{timestamp}|{nonce}|{self.user_id}|".encode("utf-8")
            + body_bytes
        )
        signature = base64.b64encode(
            hmac.new(
                self.coripo_hmac_secret.encode("utf-8"),
                base,
                hashlib.sha256,
            ).digest()
        ).decode("utf-8")

        headers = {
            "Authorization": (
                f"HMAC keyId={self.coripo_hmac_key_id}, signature={signature}"
            ),
            "X-Timestamp": timestamp,
            "X-Nonce": nonce,
            "X-User-Id": str(self.user_id),
        }
        if self.user_name:
            headers["X-User-Name"] = str(self.user_name)
        return headers

    async def _coripo_request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
        ensure_sid: bool = False,
    ) -> dict[str, Any]:
        if ensure_sid and not self._coripo_session_id and not self._can_use_coripo_hmac():
            raise ValueError(
                "Coripo request requires either a valid SID or HMAC "
                "configuration (key id + secret + user id)."
            )

        url = f"{self._coripo_api_base()}/{path.lstrip('/')}"
        headers: dict[str, str] = {"Accept": "application/json"}

        body_bytes = b""
        if json_body is not None:
            body_bytes = json.dumps(
                json_body,
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
            headers["Content-Type"] = "application/json"

        if self._coripo_session_id:
            headers["sid"] = self._coripo_session_id

        hmac_headers = self._build_coripo_hmac_headers(body_bytes)
        headers.update(hmac_headers)

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.request(
                method=method,
                url=url,
                headers=headers,
                params=params,
                content=body_bytes if json_body is not None else None,
            )

        if response.status_code >= 400:
            logger.error(
                "Coripo request failed status=%s url=%s body=%s",
                response.status_code,
                url,
                response.text,
            )
            response.raise_for_status()

        return self._decode_response(response)

    async def _bootstrap_coripo_sid(self) -> None:
        if self._coripo_session_id:
            return
        if not self._can_use_coripo_hmac():
            logger.warning(
                "Coripo SID bootstrap skipped: missing HMAC config mode=%s key_id=%s has_secret=%s has_user_id=%s",
                self.mode,
                bool(self.coripo_hmac_key_id),
                bool(self.coripo_hmac_secret),
                bool(self.user_id),
            )
            return

        # Coripo auth signs raw body. checksid works with an empty body and avoids
        # signature mismatches from serializer differences.
        try:
            data = await self._coripo_request("POST", "checksid", json_body=None)
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code != 401:
                raise

            # Retry strategy for strict Coripo auth checks:
            # 1) retry without X-User-Name
            # 2) retry with user_id from tenant token JSON (if available)
            previous_user_name = self.user_name
            previous_user_id = self.user_id

            if previous_user_name:
                self.user_name = None
                try:
                    data = await self._coripo_request("POST", "checksid", json_body=None)
                except httpx.HTTPStatusError:
                    data = None
                if data is not None:
                    sid = data.get("sid")
                    if sid and isinstance(sid, str):
                        self._coripo_session_id = sid
                        return

            fallback_user_id = self._auth_config.get("user_id")
            if fallback_user_id and str(fallback_user_id) != str(previous_user_id):
                self.user_id = str(fallback_user_id)
                self.user_name = self._auth_config.get("user_name")
                try:
                    data = await self._coripo_request("POST", "checksid", json_body=None)
                except httpx.HTTPStatusError:
                    data = None
                if data is not None:
                    sid = data.get("sid")
                    if sid and isinstance(sid, str):
                        self._coripo_session_id = sid
                        return

            self.user_id = previous_user_id
            self.user_name = previous_user_name
            raise

        sid = data.get("sid")
        if not sid and isinstance(data.get("message"), dict):
            sid = data["message"].get("sid")
        if sid and isinstance(sid, str):
            self._coripo_session_id = sid

    def _extract_coripo_global_lists(self, data: dict[str, Any]) -> list[dict[str, Any]]:
        if isinstance(data.get("lists"), list):
            return data["lists"]
        if isinstance(data.get("data"), dict) and isinstance(data["data"].get("lists"), list):
            return data["data"]["lists"]
        if isinstance(data.get("message"), dict):
            inner = data["message"].get("data")
            if isinstance(inner, dict) and isinstance(inner.get("lists"), list):
                return inner["lists"]
        return []

    def _extract_coripo_records(self, data: dict[str, Any]) -> list[dict[str, Any]]:
        if isinstance(data.get("records"), list):
            return data["records"]
        if isinstance(data.get("message"), dict):
            inner = data["message"].get("data")
            if isinstance(inner, dict) and isinstance(inner.get("records"), list):
                return inner["records"]
        if isinstance(data.get("data"), dict) and isinstance(data["data"].get("records"), list):
            return data["data"]["records"]
        return []

    def _build_coripo_list_payload(self, module_name: str, data: dict[str, Any]) -> dict[str, Any]:
        limit = int(data.get("max_results", data.get("limit", 20)) or 20)
        offset = int(data.get("offset", 0) or 0)

        payload: dict[str, Any] = {
            "offset": offset,
            "limit": max(1, min(limit, 500)),
            "savedSearch": False,
            "recordsCount": False,
            "listview_type": "list",
            "filter": {
                "operator": "and",
                "operands": [
                    {
                        "field": "*",
                        "type": "cont",
                        "value": str(data.get("q") or data.get("query") or ""),
                    }
                ],
            },
            "order": [
                {
                    "field": "date_modified",
                    "sort": "DESC",
                    "module": module_name,
                }
            ],
            "columns": {},
        }

        query = str(data.get("query") or "").strip()
        if not query:
            payload["filter"] = {
                "operator": "and",
                "operands": [{"field": "deleted", "type": "eq", "value": "0"}],
            }

        return payload

    def _sanitize_coripo_fields(self, data: dict[str, Any]) -> dict[str, Any]:
        # Coripo set/{Module} expects only module fields in "fields". Remove bridge-only keys.
        if isinstance(data.get("fields"), dict):
            source = data["fields"]
        else:
            source = data
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
        return {
            key: value
            for key, value in source.items()
            if key not in reserved and value is not None
        }

    # -----------------------------
    # Sugar v4.1 helpers
    # -----------------------------

    async def _call_v4_1(self, method: str, rest_data: dict[str, Any]) -> dict[str, Any]:
        payload = {
            "method": method,
            "input_type": "JSON",
            "response_type": "JSON",
            "rest_data": json.dumps(rest_data),
        }
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(self.base_url, data=payload)

        if response.status_code >= 400:
            logger.error(
                "SugarCRM v4_1 call failed status=%s method=%s body=%s",
                response.status_code,
                method,
                response.text,
            )
            response.raise_for_status()

        return self._decode_response(response)

    async def _get_v4_1_session(self) -> str:
        if self._session_id:
            return self._session_id

        session_id = self._auth_config.get("session_id")
        if not session_id and self.token and not self.token.startswith("{"):
            # Backward compatibility: plain token as session id.
            session_id = self.token

        if isinstance(session_id, str) and session_id.strip():
            self._session_id = session_id.strip()
            return self._session_id

        username = self._auth_config.get("username")
        password = self._auth_config.get("password")
        if not username or not password:
            raise ValueError(
                "SugarCRM v4.1 token must be either a session id or JSON "
                "with username/password"
            )

        app_name = self._auth_config.get("application_name", "sugar_voice_bridge")
        password_md5 = hashlib.md5(password.encode("utf-8")).hexdigest()
        login_payload = {
            "user_auth": {
                "user_name": username,
                "password": password_md5,
                "version": "1",
            },
            "application_name": app_name,
            "name_value_list": [],
        }
        result = await self._call_v4_1("login", login_payload)
        sid = result.get("id")
        if not sid:
            raise ValueError(f"SugarCRM v4.1 login failed: {result}")
        self._session_id = sid
        return sid

    # -----------------------------
    # Public API
    # -----------------------------

    async def generic_search(self, query: str, scope: str = "all") -> dict[str, Any]:
        scope = (scope or "all").lower().strip()

        if self.mode == "coripo_public":
            module_map = {
                "contacts": "Contacts",
                "accounts": "Accounts",
                "meetings": "Meetings",
            }
            if scope in module_map:
                module_name = module_map[scope]
                payload = self._build_coripo_list_payload(module_name, {"q": query, "max_results": 20})
                result = await self._coripo_request(
                    "POST",
                    f"list/{module_name}",
                    json_body=payload,
                    ensure_sid=True,
                )
                return {"module": module_name, "result": result}

            aggregate: dict[str, Any] = {}
            for key, module_name in module_map.items():
                payload = self._build_coripo_list_payload(module_name, {"q": query, "max_results": 20})
                aggregate[key] = await self._coripo_request(
                    "POST",
                    f"list/{module_name}",
                    json_body=payload,
                    ensure_sid=True,
                )
            return aggregate

        if self.is_v4_1:
            sid = await self._get_v4_1_session()
            escaped = self._escape_like(query)

            async def list_module(module_name: str, query_sql: str) -> dict[str, Any]:
                return await self._call_v4_1(
                    "get_entry_list",
                    {
                        "session": sid,
                        "module_name": module_name,
                        "query": query_sql,
                        "order_by": f"{module_name.lower()}.date_modified DESC",
                        "offset": 0,
                        "select_fields": [],
                        "link_name_to_fields_array": [],
                        "max_results": 20,
                        "deleted": 0,
                    },
                )

            if scope == "contacts":
                return await list_module(
                    "Contacts",
                    f"(contacts.first_name LIKE '%{escaped}%' OR contacts.last_name LIKE '%{escaped}%')",
                )
            if scope == "accounts":
                return await list_module("Accounts", f"accounts.name LIKE '%{escaped}%'")
            if scope == "meetings":
                return await list_module("Meetings", f"meetings.name LIKE '%{escaped}%'")

            return {
                "contacts": await list_module(
                    "Contacts",
                    f"(contacts.first_name LIKE '%{escaped}%' OR contacts.last_name LIKE '%{escaped}%')",
                ),
                "accounts": await list_module("Accounts", f"accounts.name LIKE '%{escaped}%'"),
                "meetings": await list_module("Meetings", f"meetings.name LIKE '%{escaped}%'"),
            }

        # Modern Sugar REST-like fallback.
        return await self._request(
            "GET",
            "/search",
            params={"q": query, "scope": scope},
        )

    async def execute_module_action(
        self,
        module: str,
        action: str,
        data: dict[str, Any],
    ) -> dict[str, Any]:
        normalized = action.lower().strip()

        if self.mode == "coripo_public":
            module_name = self._canonical_module(module)

            if normalized == "create":
                payload = {
                    "fields": self._sanitize_coripo_fields(data),
                    "relationships": data.get("relationships", {}),
                    "customData": data.get("customData", {}),
                    "files": data.get("files", []),
                    "invitees": data.get(
                        "invitees",
                        {"Users": [], "Contacts": [], "Leads": []},
                    ),
                }
                if "inviteesBackup" in data:
                    payload["inviteesBackup"] = data.get("inviteesBackup", {})
                return await self._coripo_request(
                    "POST",
                    f"set/{module_name}",
                    json_body=payload,
                    ensure_sid=True,
                )

            if normalized in {"update", "patch"}:
                record_id = data.get("id")
                if not record_id:
                    raise ValueError("Update action requires 'id' in data")
                payload = {
                    "fields": self._sanitize_coripo_fields(data),
                    "relationships": data.get("relationships", {}),
                    "customData": data.get("customData", {}),
                    "files": data.get("files", []),
                    "invitees": data.get(
                        "invitees",
                        {"Users": [], "Contacts": [], "Leads": []},
                    ),
                    "inviteesBackup": data.get(
                        "inviteesBackup",
                        data.get("invitees", {"Users": [], "Contacts": [], "Leads": []}),
                    ),
                }
                return await self._coripo_request(
                    "POST",
                    f"set/{module_name}/{record_id}",
                    json_body=payload,
                    ensure_sid=True,
                )

            if normalized == "delete":
                record_id = data.get("id")
                if not record_id:
                    raise ValueError("Delete action requires 'id' in data")
                return await self._coripo_request(
                    "DELETE",
                    f"delete/{module_name}/{record_id}",
                    ensure_sid=True,
                )

            if normalized in {"get", "read", "detail"}:
                record_id = data.get("id")
                if not record_id:
                    raise ValueError("Read action requires 'id' in data")
                return await self._coripo_request(
                    "GET",
                    f"detail/{module_name}/{record_id}",
                    ensure_sid=True,
                )

            if normalized in {"list", "search"}:
                payload = self._build_coripo_list_payload(module_name, data)
                return await self._coripo_request(
                    "POST",
                    f"list/{module_name}",
                    json_body=payload,
                    ensure_sid=True,
                )

        if self.is_v4_1:
            module_name = self._canonical_module(module)
            sid = await self._get_v4_1_session()

            if normalized == "create":
                values = [
                    {"name": k, "value": v}
                    for k, v in data.items()
                    if k != "id" and v is not None
                ]
                return await self._call_v4_1(
                    "set_entry",
                    {"session": sid, "module_name": module_name, "name_value_list": values},
                )

            if normalized in {"update", "patch"}:
                if not data.get("id"):
                    raise ValueError("Update action requires 'id' in data")
                values = [{"name": "id", "value": data["id"]}] + [
                    {"name": k, "value": v}
                    for k, v in data.items()
                    if k != "id" and v is not None
                ]
                return await self._call_v4_1(
                    "set_entry",
                    {"session": sid, "module_name": module_name, "name_value_list": values},
                )

            if normalized == "delete":
                record_id = data.get("id")
                if not record_id:
                    raise ValueError("Delete action requires 'id' in data")
                return await self._call_v4_1(
                    "set_entry",
                    {
                        "session": sid,
                        "module_name": module_name,
                        "name_value_list": [
                            {"name": "id", "value": record_id},
                            {"name": "deleted", "value": 1},
                        ],
                    },
                )

            if normalized in {"get", "read", "detail"}:
                record_id = data.get("id")
                if not record_id:
                    raise ValueError("Read action requires 'id' in data")
                return await self._call_v4_1(
                    "get_entry",
                    {
                        "session": sid,
                        "module_name": module_name,
                        "id": record_id,
                        "select_fields": data.get("select_fields", []),
                        "link_name_to_fields_array": [],
                    },
                )

            if normalized in {"list", "search"}:
                query_sql = data.get("query", "")
                return await self._call_v4_1(
                    "get_entry_list",
                    {
                        "session": sid,
                        "module_name": module_name,
                        "query": query_sql,
                        "order_by": data.get("order_by", f"{module_name.lower()}.date_modified DESC"),
                        "offset": int(data.get("offset", 0)),
                        "select_fields": data.get("select_fields", []),
                        "link_name_to_fields_array": [],
                        "max_results": int(data.get("max_results", 20)),
                        "deleted": int(data.get("deleted", 0)),
                    },
                )

        module_name = module.strip("/")
        if normalized == "create":
            return await self._request("POST", f"/{module_name}", json_body=data)
        if normalized in {"update", "patch"}:
            record_id = data.get("id")
            if not record_id:
                raise ValueError("Update action requires 'id' in data")
            body = {k: v for k, v in data.items() if k != "id"}
            return await self._request("PUT", f"/{module_name}/{record_id}", json_body=body)
        if normalized == "delete":
            record_id = data.get("id")
            if not record_id:
                raise ValueError("Delete action requires 'id' in data")
            return await self._request("DELETE", f"/{module_name}/{record_id}")
        if normalized in {"get", "read", "detail"}:
            record_id = data.get("id")
            if not record_id:
                raise ValueError("Read action requires 'id' in data")
            return await self._request("GET", f"/{module_name}/{record_id}")
        if normalized in {"list", "search"}:
            return await self._request("GET", f"/{module_name}", params=data)

        return await self._request(
            normalized.upper(),
            f"/{module_name}/{normalized}",
            json_body=data,
        )

    async def get_dynamic_schema(self, module: str) -> dict[str, Any]:
        module_name = self._canonical_module(module)

        if self.mode == "coripo_public":
            return await self._coripo_request(
                "GET",
                f"defs/{module_name}",
                ensure_sid=True,
            )

        if self.is_v4_1:
            sid = await self._get_v4_1_session()
            return await self._call_v4_1(
                "get_module_fields",
                {
                    "session": sid,
                    "module_name": module_name,
                    "fields": [],
                },
            )

        try:
            return await self._request(
                "GET",
                "/metadata",
                params={"type_filter": "modules", "module_filter[]": module_name},
            )
        except httpx.HTTPError:
            logger.warning("Metadata endpoint unavailable, falling back to AiIngest schema")
            return await self._request("GET", f"/AiIngest/schema/{module_name}")

    async def fetch_ai_ingest_dump(self, module: str) -> dict[str, Any]:
        module_name = self._canonical_module(module)

        if self.mode == "coripo_public":
            payload = self._build_coripo_list_payload(module_name, {"max_results": 500})
            data = await self._coripo_request(
                "POST",
                f"list/{module_name}",
                json_body=payload,
                ensure_sid=True,
            )
            return {"records": self._extract_coripo_records(data)}

        if not self.is_v4_1:
            return await self._request("GET", f"/AiIngest/dump/{module_name}")

        parsed = urlparse(self.base_url)
        host_root = f"{parsed.scheme}://{parsed.netloc}"
        url = f"{host_root}/AiIngest/dump/{module_name}"
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(url)
        if response.status_code >= 400:
            logger.error(
                "AiIngest dump failed status=%s url=%s body=%s",
                response.status_code,
                url,
                response.text,
            )
            response.raise_for_status()
        return self._decode_response(response)
