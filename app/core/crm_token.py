"""Parsing of a tenant's stored CRM token (the single source of per-tenant HMAC config).

A tenant's HMAC key id + secret live only in its encrypted DB token. The same pair
signs gateway -> Coripo calls and verifies Coripo -> gateway requests.
"""
from __future__ import annotations

import json
from typing import Any


def parse_crm_token(token: str) -> dict[str, Any]:
    """Return the auth config dict encoded in a CRM token, or {} if it has none.

    Accepted shapes: JSON object (``hmac_key_id``/``hmac_secret``/``session_id``/...),
    single-entry ``{"keyId": "secret"}``, or compact ``keyId:secret``.
    """
    token = (token or "").strip()
    if not token:
        return {}
    if token.startswith("{"):
        try:
            parsed = json.loads(token)
        except json.JSONDecodeError:
            return {}
        if not isinstance(parsed, dict):
            return {}
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
    if ":" in token:
        key_id, secret = token.split(":", 1)
        if key_id.strip() and secret.strip():
            return {
                "mode": "coripo_public",
                "hmac_key_id": key_id.strip(),
                "hmac_secret": secret.strip(),
            }
    return {}


def build_crm_token(key_id: str, secret: str) -> str:
    """Canonical token stored for a tenant (what the onboarding wizard writes)."""
    return json.dumps(
        {"mode": "coripo_public", "hmac_key_id": key_id, "hmac_secret": secret},
        separators=(",", ":"),
    )
