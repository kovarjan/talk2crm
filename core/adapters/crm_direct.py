import base64
import hashlib
import hmac
import json
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import requests

from core.clients.client_loader import get_rest_client_config, RestClientConfig
from core.config import CRM_DIRECT_BACKEND


MODULE_MAP = {
    "meetings": "Meetings",
    "contacts": "Contacts",
    "tasks": "Tasks",
    "calls": "Calls",
    "notes": "Notes",
    "accounts": "Accounts",
    "companies": "Accounts",
    "leads": "Leads",
    "users": "Users",
}


class CrmDirectError(RuntimeError):
    pass


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _sign(secret: str, ts: str, nonce: str, user_id: str, body: bytes) -> str:
    base = f"{ts}|{nonce}|{user_id}|".encode("utf-8") + body
    mac = hmac.new(secret.encode("utf-8"), base, hashlib.sha256).digest()
    return base64.b64encode(mac).decode("ascii")


def _normalize_module(module: str) -> str:
    if not module:
        raise ValueError("Missing CRM module")
    key = str(module).strip()
    if not key:
        raise ValueError("Missing CRM module")
    mapped = MODULE_MAP.get(key.lower())
    return mapped or key


def _strip_nones(payload: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in payload.items() if v is not None}


def _merge_metadata(payload: Dict[str, Any], metadata: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in (metadata or {}).items():
        if v is None:
            continue
        payload.setdefault(k, v)
    return payload


def _build_meeting_payload(parameters: Dict[str, Any], metadata: Dict[str, Any]) -> Dict[str, Any]:
    payload: Dict[str, Any] = {}
    params = parameters or {}
    meta = metadata or {}

    payload["name"] = params.get("name")

    related_module = params.get("related_module")
    related_to_id = params.get("related_to_id")
    if related_to_id:
        payload["parent_id"] = related_to_id
        if related_module:
            payload["parent_type"] = _normalize_module(related_module)

    date = meta.get("date")
    time = meta.get("time")
    if meta.get("date_start"):
        payload["date_start"] = meta.get("date_start")
    elif date and time:
        payload["date_start"] = f"{date} {time}:00"
    elif date:
        payload["date_start"] = date

    duration = meta.get("duration")
    if duration is not None:
        try:
            total_minutes = int(duration)
            payload["duration_hours"] = total_minutes // 60
            payload["duration_minutes"] = total_minutes % 60
        except (TypeError, ValueError):
            pass

    if meta.get("location"):
        payload["location"] = meta.get("location")

    payload = _merge_metadata(payload, meta)
    return _strip_nones(payload)


def _build_payload(command: Dict[str, Any]) -> Dict[str, Any]:
    params = command.get("parameters") or {}
    meta = command.get("metadata") or {}
    module = (command.get("module") or "").lower().strip()

    if module == "meetings":
        return _build_meeting_payload(params, meta)

    payload: Dict[str, Any] = dict(params)
    if module == "contacts" and meta.get("notes") and not payload.get("description"):
        payload["description"] = meta.get("notes")

    payload = _merge_metadata(payload, meta)
    return _strip_nones(payload)


# ------------------------- Coripo custom REST backend -------------------------

def _request(
    method: str,
    base_url: str,
    path: str,
    payload: Optional[Dict[str, Any]] = None,
    key_id: Optional[str] = None,
    secret: Optional[str] = None,
    user_id: Optional[str] = None,
    user_name: Optional[str] = None,
    tenant: Optional[str] = None,
    user_id_header: str = "X-User-Id",
    user_name_header: str = "X-User-Name",
    tenant_header: str = "X-Tenant",
) -> Dict[str, Any]:
    body = b""
    if payload is not None:
        body = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")

    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "X-Request-Id": str(uuid.uuid4()),
    }

    if user_id and user_id_header:
        headers[user_id_header] = str(user_id)
    if user_name and user_name_header:
        headers[user_name_header] = str(user_name)
    if tenant and tenant_header:
        headers[tenant_header] = str(tenant)

    if not user_id:
        raise CrmDirectError("Missing user_id for HMAC signature")
    if not key_id or not secret:
        raise CrmDirectError("Missing CRM REST HMAC credentials")
    ts = _now_iso()
    nonce = str(uuid.uuid4())
    sig = _sign(secret, ts, nonce, str(user_id), body)
    headers.update(
        {
            "Authorization": f"HMAC keyId={key_id}, signature={sig}",
            "X-Timestamp": ts,
            "X-Nonce": nonce,
            "Idempotency-Key": str(uuid.uuid4()),
        }
    )

    url = base_url.rstrip("/") + "/" + path.lstrip("/")
    r = requests.request(method, url, headers=headers, data=body if body else None, timeout=20)
    ct = r.headers.get("content-type", "")
    text = r.text

    try:
        r.raise_for_status()
    except requests.HTTPError as e:
        snippet = (text or "")[:800]
        raise CrmDirectError(f"CRM HTTP {r.status_code} at {url}: {snippet}") from e

    if "application/json" in ct.lower():
        try:
            return r.json()
        except ValueError:
            pass

    return {"status": r.status_code, "raw": text}


class CrmDirectClient:
    def __init__(self, tenant: str) -> None:
        cfg = get_rest_client_config(tenant)
        self.base_url = cfg.rest_url
        self.key_id = cfg.rest_hmac_key_id
        self.secret = cfg.rest_hmac_secret
        self.user_id_header = cfg.rest_user_id_header
        self.user_name_header = cfg.rest_user_name_header
        self.tenant_header = cfg.rest_tenant_header

    def request(
        self,
        method: str,
        path: str,
        payload: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        user_name: Optional[str] = None,
        tenant: Optional[str] = None,
    ) -> Dict[str, Any]:
        return _request(
            method,
            self.base_url,
            path,
            payload=payload,
            key_id=self.key_id,
            secret=self.secret,
            user_id=user_id,
            user_name=user_name,
            tenant=tenant,
            user_id_header=self.user_id_header,
            user_name_header=self.user_name_header,
            tenant_header=self.tenant_header,
        )


# ---------------------------- Sugar v4.1 backend -----------------------------

def _md5_hex(value: str) -> str:
    return hashlib.md5(value.encode("utf-8")).hexdigest()  # nosec B324 (Sugar API contract)


def _to_name_value_list(payload: Dict[str, Any]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for k, v in (payload or {}).items():
        if isinstance(v, (dict, list)):
            v = json.dumps(v, ensure_ascii=False)
        out.append({"name": str(k), "value": "" if v is None else str(v)})
    return out


def _name_value_list_to_dict(nvl: Any) -> Dict[str, Any]:
    if isinstance(nvl, dict):
        out: Dict[str, Any] = {}
        for k, v in nvl.items():
            if isinstance(v, dict) and "value" in v:
                out[k] = v.get("value")
            else:
                out[k] = v
        return out
    if not isinstance(nvl, list):
        return {}
    out = {}
    for item in nvl:
        if isinstance(item, dict):
            name = item.get("name")
            if name:
                out[str(name)] = item.get("value")
    return out


def _normalize_sugar_list_response(data: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(data, dict):
        return data

    if isinstance(data.get("entry_list"), list):
        records = []
        for raw in data["entry_list"]:
            if not isinstance(raw, dict):
                continue
            row = _name_value_list_to_dict(raw.get("name_value_list"))
            if raw.get("id") and "id" not in row:
                row["id"] = raw.get("id")
            if raw.get("module_name") and "module" not in row:
                row["module"] = raw.get("module_name")
            records.append(row)
        return {
            "module": data.get("module_name"),
            "records": records,
            "result_count": data.get("result_count"),
            "next_offset": data.get("next_offset"),
        }

    if isinstance(data.get("name_value_list"), (list, dict)):
        normalized = _name_value_list_to_dict(data["name_value_list"])
        if data.get("id") and "id" not in normalized:
            normalized["id"] = data.get("id")
        return normalized

    return data


def _sugar_request(rest_url: str, method: str, rest_data: Dict[str, Any], timeout: int = 20) -> Dict[str, Any]:
    payload = {
        "method": method,
        "input_type": "JSON",
        "response_type": "JSON",
        "rest_data": json.dumps(rest_data, separators=(",", ":"), ensure_ascii=False),
    }
    r = requests.post(rest_url, data=payload, timeout=timeout)
    text = r.text
    try:
        r.raise_for_status()
    except requests.HTTPError as e:
        snippet = (text or "")[:800]
        raise CrmDirectError(f"Sugar REST HTTP {r.status_code} at {rest_url}: {snippet}") from e

    try:
        data = r.json()
    except ValueError as e:
        snippet = (text or "")[:800]
        raise CrmDirectError(f"Sugar REST non-JSON response at {rest_url}: {snippet}") from e

    if isinstance(data, dict) and data.get("name") in ("Invalid Login", "Access Denied"):
        msg = data.get("description") or data.get("name")
        raise CrmDirectError(f"Sugar REST auth error: {msg}")

    return data


def _resolve_session_with_hmac(cfg: RestClientConfig, user_id: str, user_name: Optional[str]) -> str:
    if not cfg.sugar_session_resolver_url:
        raise CrmDirectError("Missing sugar_session_resolver_url for Sugar resolver auth mode")

    body_obj = {"user_id": user_id}
    if user_name:
        body_obj["user_name"] = user_name
    body = json.dumps(body_obj, separators=(",", ":"), ensure_ascii=False).encode("utf-8")

    ts = _now_iso()
    nonce = str(uuid.uuid4())
    sig = _sign(cfg.rest_hmac_secret, ts, nonce, user_id, body)

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"HMAC keyId={cfg.rest_hmac_key_id}, signature={sig}",
        "X-Timestamp": ts,
        "X-Nonce": nonce,
        cfg.rest_user_id_header: user_id,
        "X-Request-Id": str(uuid.uuid4()),
    }
    if user_name and cfg.rest_user_name_header:
        headers[cfg.rest_user_name_header] = user_name

    method = (cfg.sugar_session_resolver_method or "POST").upper().strip()
    if method not in {"POST", "GET"}:
        raise CrmDirectError(f"Unsupported sugar_session_resolver_method: {method}")

    if method == "GET":
        resp = requests.get(cfg.sugar_session_resolver_url, headers=headers, params=body_obj, timeout=20)
    else:
        resp = requests.post(cfg.sugar_session_resolver_url, headers=headers, data=body, timeout=20)

    text = resp.text
    try:
        resp.raise_for_status()
    except requests.HTTPError as e:
        snippet = (text or "")[:800]
        raise CrmDirectError(f"Session resolver HTTP {resp.status_code}: {snippet}") from e

    try:
        data = resp.json()
    except ValueError as e:
        snippet = (text or "")[:800]
        raise CrmDirectError(f"Session resolver non-JSON response: {snippet}") from e

    sid = ""
    if isinstance(data, dict):
        sid = str(
            data.get("sid")
            or data.get("session")
            or data.get("session_id")
            or data.get("id")
            or ""
        ).strip()

    if not sid:
        raise CrmDirectError("Session resolver did not return sid/session_id")
    return sid


def _sugar_login(cfg: RestClientConfig) -> str:
    if not cfg.sugar_username or not cfg.sugar_password:
        raise CrmDirectError("Missing sugar_username/sugar_password for Sugar login mode")

    password = cfg.sugar_password if cfg.sugar_password_is_md5 else _md5_hex(cfg.sugar_password)
    rest_data = {
        "user_auth": {
            "user_name": cfg.sugar_username,
            "password": password,
            "version": "1",
        },
        "application_name": cfg.sugar_application_name or "talk2api",
        "name_value_list": [],
    }
    out = _sugar_request(cfg.sugar_rest_url, "login", rest_data)
    session_id = str((out or {}).get("id") or "").strip()
    if not session_id:
        raise CrmDirectError("Sugar login did not return session id")
    return session_id


def _get_sugar_session(cfg: RestClientConfig, user_id: Optional[str], user_name: Optional[str]) -> str:
    mode = (cfg.sugar_auth_mode or "login").lower().strip()

    if mode in {"sid", "session", "session_id"}:
        sid = (cfg.sugar_session_id or "").strip()
        if not sid:
            raise CrmDirectError("Missing sugar_session_id for Sugar session auth mode")
        return sid

    if mode in {"resolver", "hmac_resolver", "user_hmac"}:
        if not user_id:
            raise CrmDirectError("Missing user_id for resolver-based Sugar auth mode")
        return _resolve_session_with_hmac(cfg, str(user_id), user_name)

    # default login mode
    return _sugar_login(cfg)


def _sql_quote(v: Any) -> str:
    if v is None:
        return "NULL"
    s = str(v).replace("'", "\\'")
    return f"'{s}'"


def _sugar_filter_to_query(filter_value: Any) -> str:
    if not filter_value:
        return ""
    if isinstance(filter_value, str):
        return filter_value

    if not isinstance(filter_value, dict):
        return ""

    operator = str(filter_value.get("operator") or "").lower().strip()
    operands = filter_value.get("operands") or []

    if operator and isinstance(operands, list) and len(operands) >= 2:
        field = str(operands[0])
        value = operands[1]

        if operator == "eq":
            return f"{field} = {_sql_quote(value)}"
        if operator == "neq":
            return f"{field} != {_sql_quote(value)}"
        if operator == "cont":
            return f"{field} LIKE {_sql_quote('%' + str(value) + '%')}"
        if operator == "startswith":
            return f"{field} LIKE {_sql_quote(str(value) + '%')}"
        if operator == "endswith":
            return f"{field} LIKE {_sql_quote('%' + str(value))}"
        if operator == "gt":
            return f"{field} > {_sql_quote(value)}"
        if operator == "gte":
            return f"{field} >= {_sql_quote(value)}"
        if operator == "lt":
            return f"{field} < {_sql_quote(value)}"
        if operator == "lte":
            return f"{field} <= {_sql_quote(value)}"
        if operator == "in" and isinstance(value, list):
            vals = ",".join(_sql_quote(x) for x in value)
            return f"{field} IN ({vals})"
        if operator == "nin" and isinstance(value, list):
            vals = ",".join(_sql_quote(x) for x in value)
            return f"{field} NOT IN ({vals})"

    clauses = []
    for key, value in filter_value.items():
        if isinstance(value, (dict, list)):
            continue
        clauses.append(f"{key} = {_sql_quote(value)}")
    return " AND ".join(clauses)


def _sugar_order_by(order: Any, order_by: Any) -> str:
    if isinstance(order_by, str) and order_by.strip():
        return order_by.strip()

    if isinstance(order, str) and order.strip():
        return order.strip()

    if isinstance(order, dict):
        field = str(order.get("field") or order.get("name") or "").strip()
        direction = str(order.get("direction") or order.get("dir") or "ASC").upper().strip()
        if field:
            if direction not in {"ASC", "DESC"}:
                direction = "ASC"
            return f"{field} {direction}"

    return ""


def _execute_sugar_command(
    command: Dict[str, Any],
    cfg: RestClientConfig,
    user_id: Optional[str],
    user_name: Optional[str],
) -> Dict[str, Any]:
    if not cfg.sugar_rest_url:
        raise CrmDirectError("Missing sugar_rest_url for Sugar v4.1 backend")

    action = (command.get("action") or "").lower().strip()
    module = _normalize_module(command.get("module") or "")
    record_id = command.get("updateId") or (command.get("parameters") or {}).get("id")
    payload = _build_payload(command)

    session = _get_sugar_session(cfg, user_id=user_id, user_name=user_name)

    if action in ("create", "set"):
        out = _sugar_request(
            cfg.sugar_rest_url,
            "set_entry",
            {
                "session": session,
                "module_name": module,
                "name_value_list": _to_name_value_list(payload),
            },
        )
        return _normalize_sugar_list_response(out)

    if action in ("update",):
        if not record_id:
            raise CrmDirectError("Missing record id for update")
        payload = dict(payload)
        payload["id"] = record_id
        out = _sugar_request(
            cfg.sugar_rest_url,
            "set_entry",
            {
                "session": session,
                "module_name": module,
                "name_value_list": _to_name_value_list(payload),
            },
        )
        return _normalize_sugar_list_response(out)

    if action in ("delete", "remove"):
        if not record_id:
            raise CrmDirectError("Missing record id for delete")
        out = _sugar_request(
            cfg.sugar_rest_url,
            "set_entry",
            {
                "session": session,
                "module_name": module,
                "name_value_list": _to_name_value_list({"id": record_id, "deleted": "1"}),
            },
        )
        return _normalize_sugar_list_response(out)

    if action in ("get", "read", "detail"):
        if not record_id:
            raise CrmDirectError("Missing record id for detail")
        out = _sugar_request(
            cfg.sugar_rest_url,
            "get_entry",
            {
                "session": session,
                "module_name": module,
                "id": record_id,
                "select_fields": [],
                "link_name_to_fields_array": [],
            },
        )
        normalized = _normalize_sugar_list_response(out)
        if isinstance(normalized, dict) and isinstance(normalized.get("records"), list):
            return normalized["records"][0] if normalized["records"] else {}
        return normalized

    if action in ("list", "search"):
        where = payload.get("where")
        if where is None:
            where = _sugar_filter_to_query(payload.get("filter"))
        order_by = _sugar_order_by(payload.get("order"), payload.get("orderBy"))
        max_results = int(payload.get("limit", 100))
        offset = int(payload.get("offset", 0))
        select_fields = payload.get("select_fields") if isinstance(payload.get("select_fields"), list) else []

        out = _sugar_request(
            cfg.sugar_rest_url,
            "get_entry_list",
            {
                "session": session,
                "module_name": module,
                "query": where or "",
                "order_by": order_by,
                "offset": offset,
                "select_fields": select_fields,
                "link_name_to_fields_array": [],
                "max_results": max_results,
                "deleted": 0,
                "favorites": False,
            },
        )
        normalized = _normalize_sugar_list_response(out)
        if isinstance(normalized, dict) and "module" not in normalized:
            normalized["module"] = module
        return normalized

    raise CrmDirectError(f"Unsupported CRM action: {action}")


def _backend_for_tenant(tenant: str) -> str:
    cfg = get_rest_client_config(tenant)
    backend = (cfg.crm_direct_backend or CRM_DIRECT_BACKEND or "coripo").lower().strip()
    if backend in {"legacy", "sugar", "sugarcrm", "sugar_v4", "sugar_v4_1"}:
        return "sugar_v4_1"
    return "coripo"


def execute_direct_command(command: Dict[str, Any], user_id: Optional[str] = None, tenant: Optional[str] = None, user_name: Optional[str] = None) -> Dict[str, Any]:
    if not tenant:
        raise CrmDirectError("Missing tenant for CRM REST call")

    cfg = get_rest_client_config(tenant)
    backend = _backend_for_tenant(tenant)
    if backend == "sugar_v4_1":
        return _execute_sugar_command(command, cfg=cfg, user_id=user_id, user_name=user_name)

    client = CrmDirectClient(tenant)
    action = (command.get("action") or "").lower().strip()
    module = _normalize_module(command.get("module") or "")

    record_id = command.get("updateId") or (command.get("parameters") or {}).get("id")
    payload = _build_payload(command)

    if action in ("create", "set"):
        return client.request("POST", f"/set/{module}", payload=payload, user_id=user_id, user_name=user_name, tenant=tenant)
    if action in ("update",):
        if not record_id:
            raise CrmDirectError("Missing record id for update")
        return client.request("POST", f"/set/{module}/{record_id}", payload=payload, user_id=user_id, user_name=user_name, tenant=tenant)
    if action in ("delete", "remove"):
        if not record_id:
            raise CrmDirectError("Missing record id for delete")
        return client.request("DELETE", f"/delete/{module}/{record_id}", user_id=user_id, user_name=user_name, tenant=tenant)
    if action in ("get", "read", "detail"):
        if not record_id:
            raise CrmDirectError("Missing record id for detail")
        return client.request("GET", f"/detail/{module}/{record_id}", user_id=user_id, user_name=user_name, tenant=tenant)
    if action in ("list", "search"):
        return client.request("POST", f"/list/{module}", payload=payload, user_id=user_id, user_name=user_name, tenant=tenant)

    raise CrmDirectError(f"Unsupported CRM action: {action}")


def get_module_template(module: str, tenant: str, user_id: str, user_name: Optional[str] = None) -> Dict[str, Any]:
    backend = _backend_for_tenant(tenant)
    normalized = _normalize_module(module)

    if backend == "sugar_v4_1":
        cfg = get_rest_client_config(tenant)
        session = _get_sugar_session(cfg, user_id=user_id, user_name=user_name)
        out = _sugar_request(
            cfg.sugar_rest_url,
            "get_module_fields",
            {
                "session": session,
                "module_name": normalized,
                "fields": [],
            },
        )
        return _normalize_sugar_list_response(out)

    client = CrmDirectClient(tenant)
    return client.request(
        "POST",
        f"/detail/{normalized}",
        payload={},
        user_id=user_id,
        user_name=user_name,
        tenant=tenant,
    )


def get_quickform_template(module: str, tenant: str, user_id: str, user_name: Optional[str] = None) -> Dict[str, Any]:
    backend = _backend_for_tenant(tenant)
    normalized = _normalize_module(module)

    if backend == "sugar_v4_1":
        # Sugar 6.5 v4.1 does not have quickform; return module field metadata.
        return get_module_template(normalized, tenant=tenant, user_id=user_id, user_name=user_name)

    client = CrmDirectClient(tenant)
    return client.request(
        "POST",
        f"/quickform/{normalized}",
        payload={},
        user_id=user_id,
        user_name=user_name,
        tenant=tenant,
    )


def get_available_modules(tenant: str, user_id: str, user_name: Optional[str] = None, device: str = "desktop") -> Dict[str, Any]:
    backend = _backend_for_tenant(tenant)
    if backend == "sugar_v4_1":
        cfg = get_rest_client_config(tenant)
        session = _get_sugar_session(cfg, user_id=user_id, user_name=user_name)
        out = _sugar_request(cfg.sugar_rest_url, "get_available_modules", {"session": session})
        return _normalize_sugar_list_response(out)

    cfg = get_rest_client_config(tenant)
    path = f"/ai/v1/modules?user_id={user_id}&user_name={user_name or ''}&device={device}"
    return _request(
        "GET",
        cfg.rest_url,
        path,
        payload=None,
        key_id=cfg.rest_hmac_key_id,
        secret=cfg.rest_hmac_secret,
        user_id=user_id,
        user_name=user_name,
        tenant=tenant,
        user_id_header=cfg.rest_user_id_header,
        user_name_header=cfg.rest_user_name_header,
        tenant_header=cfg.rest_tenant_header,
    )
