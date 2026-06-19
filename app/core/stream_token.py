from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
import uuid
from typing import Any


def create_stream_token(
    *,
    tenant: str,
    user_id: str,
    user_name: str,
    key_id: str,
    secret: str,
) -> str:
    """Create a 60-second HMAC-signed identity token for stream endpoint auth."""
    payload = {
        "tenant": tenant,
        "user_id": user_id,
        "user_name": user_name,
        "key_id": key_id,
        "exp": int(time.time()) + 60,
        "nonce": uuid.uuid4().hex,
    }
    payload_b64 = base64.b64encode(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).decode("utf-8")
    sig = hmac.new(
        secret.encode("utf-8"),
        payload_b64.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    return f"{payload_b64}.{sig}"


def validate_stream_token(token: str, *, hmac_keys: dict[str, str]) -> dict[str, Any]:
    """
    Validate a stream token. Returns identity payload on success.
    Raises ValueError on invalid/expired/replayed token.
    """
    try:
        payload_b64, sig = token.rsplit(".", 1)
    except ValueError:
        raise ValueError("Malformed token: missing signature separator")

    try:
        payload = json.loads(base64.b64decode(payload_b64).decode("utf-8"))
    except Exception:
        raise ValueError("Malformed token: cannot decode payload")

    key_id = str(payload.get("key_id", ""))
    secret = hmac_keys.get(key_id)
    if not secret:
        raise ValueError(f"Unknown key_id in stream token: {key_id!r}")

    expected = hmac.new(
        secret.encode("utf-8"),
        payload_b64.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    if not hmac.compare_digest(sig, expected):
        raise ValueError("Invalid token signature")

    if int(time.time()) > int(payload.get("exp", 0)):
        raise ValueError("Token expired")

    from app.core.security import _register_nonce

    nonce = str(payload.get("nonce", ""))
    if not _register_nonce(f"stream:{key_id}", nonce, 120.0):
        raise ValueError("Token nonce already used (replay attack)")

    return {
        "tenant": str(payload.get("tenant", "")),
        "user_id": str(payload.get("user_id", "")),
        "user_name": str(payload.get("user_name", "") or ""),
    }
