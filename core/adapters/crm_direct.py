import base64
import hashlib
import hmac
import json
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import requests

from core.clients.client_loader import get_rest_client_config


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


def execute_direct_command(command: Dict[str, Any], user_id: Optional[str] = None, tenant: Optional[str] = None, user_name: Optional[str] = None) -> Dict[str, Any]:
    if not tenant:
        raise CrmDirectError("Missing tenant for CRM REST call")
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
    client = CrmDirectClient(tenant)
    normalized = _normalize_module(module)
    return client.request(
        "POST",
        f"/detail/{normalized}",
        payload={},
        user_id=user_id,
        user_name=user_name,
        tenant=tenant,
    )


def get_quickform_template(module: str, tenant: str, user_id: str, user_name: Optional[str] = None) -> Dict[str, Any]:
    client = CrmDirectClient(tenant)
    normalized = _normalize_module(module)
    return client.request(
        "POST",
        f"/quickform/{normalized}",
        payload={},
        user_id=user_id,
        user_name=user_name,
        tenant=tenant,
    )
