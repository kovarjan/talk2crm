from __future__ import annotations

import base64
import hashlib
import hmac
import re
import time
from collections.abc import Mapping
from dataclasses import dataclass

from cryptography.fernet import Fernet, InvalidToken

from app.core.config import get_settings
from app.core.logging import get_logger
from app.core.nonce_store import get_memory_store, get_nonce_store


logger = get_logger(__name__)
_WARNED: set[str] = set()


def _warn_once(key: str, message: str) -> None:
    if key not in _WARNED:
        _WARNED.add(key)
        logger.warning(message)


def _register_nonce(key_id: str, nonce: str, ttl_seconds: float) -> bool:
    """Process-local replay check (sync callers, tests, dev). Production
    request paths use the shared store via ``verify_hmac_request_async``."""
    return get_memory_store().register_sync(key_id, nonce, ttl_seconds)


HMAC_AUTH_PATTERN = re.compile(
    r"^HMAC\s+keyId=(?P<kid>[^,]+),\s*signature=(?P<sig>.+)$",
    re.IGNORECASE,
)


@dataclass
class ParsedHmacHeader:
    key_id: str
    signature_b64: str


def parse_hmac_authorization(value: str) -> ParsedHmacHeader | None:
    if not value:
        return None
    match = HMAC_AUTH_PATTERN.match(value.strip())
    if not match:
        return None
    return ParsedHmacHeader(
        key_id=match.group("kid").strip(),
        signature_b64=match.group("sig").strip(),
    )


def _derive_fernet(secret: str) -> Fernet:
    try:
        return Fernet(secret.encode("utf-8"))
    except Exception:
        digest = hashlib.sha256(secret.encode("utf-8")).digest()
        key = base64.urlsafe_b64encode(digest)
        return Fernet(key)


def encrypt_tenant_secret(plain_text: str) -> str:
    settings = get_settings()
    fernet = _derive_fernet(settings.tenant_secret_key)
    return fernet.encrypt(plain_text.encode("utf-8")).decode("utf-8")


def decrypt_tenant_secret(cipher_text: str) -> str:
    settings = get_settings()
    fernet = _derive_fernet(settings.tenant_secret_key)
    try:
        return fernet.decrypt(cipher_text.encode("utf-8")).decode("utf-8")
    except InvalidToken as exc:
        raise ValueError("Unable to decrypt tenant secret") from exc


def _signing_inputs(
    *,
    method: str,
    candidate_path: str,
    timestamp: str,
    nonce: str,
    body_text: str,
    query: str,
    tenant: str,
    user_id: str,
) -> dict[str, str]:
    """Legacy string: method, path, ts, nonce, body (what the PHP client signs
    today). Bound string additionally covers query, tenant and user id; the
    variable-free fields come before the body so the layout is unambiguous."""
    legacy = "\n".join([method.upper(), candidate_path, timestamp, nonce, body_text])
    bound = "\n".join(
        [method.upper(), candidate_path, timestamp, nonce, query, tenant, user_id, body_text]
    )
    return {"legacy": legacy, "bound": bound}


def _verify_signature(
    *,
    method: str,
    path: str,
    original_path: str | None,
    body: bytes,
    authorization: str | None,
    timestamp: str | None,
    nonce: str | None,
    keys: Mapping[str, str],
    query: str = "",
    tenant: str | None = None,
    user_id: str | None = None,
) -> ParsedHmacHeader | None:
    """Check signature and skew against ``keys`` (key id -> secret, loaded from the
    tenant named in the request, so a key can only act for its own tenant).
    Does NOT touch the nonce store."""
    settings = get_settings()
    parsed = parse_hmac_authorization(authorization or "")
    if not parsed:
        return None
    if not timestamp or not nonce:
        return None

    try:
        ts = int(timestamp)
    except ValueError:
        return None

    now = int(time.time())
    if abs(now - ts) > settings.hmac_max_skew_seconds:
        return None

    key = keys.get(parsed.key_id)
    if not key:
        return None

    bind_identity = bool(getattr(settings, "hmac_bind_identity", False))
    if bind_identity and (tenant is None or user_id is None):
        return None

    body_text = body.decode("utf-8", errors="replace")
    candidate_paths: list[str] = []
    for candidate in [path, original_path]:
        value = (candidate or "").strip()
        if value and value not in candidate_paths:
            candidate_paths.append(value)

    # Some reverse proxies rewrite /v1/... to /... before forwarding.
    # Try the prefixed path as a second candidate; if neither matches we
    # return None — we do NOT fall through with a partial match.
    if path and not path.startswith("/v1/"):
        prefixed = f"/v1{path}"
        if prefixed not in candidate_paths:
            candidate_paths.append(prefixed)

    # In bind mode only the identity-bound string is accepted. Otherwise both
    # are, so gateway and clients can be upgraded one after the other.
    formats = ["bound"] if bind_identity else ["bound", "legacy"]
    for candidate_path in candidate_paths:
        inputs = _signing_inputs(
            method=method,
            candidate_path=candidate_path,
            timestamp=timestamp,
            nonce=nonce,
            body_text=body_text,
            query=query,
            tenant=tenant or "",
            user_id=user_id or "",
        )
        for fmt in formats:
            if fmt == "bound" and tenant is None and user_id is None:
                continue
            expected = hmac.new(
                key.encode("utf-8"),
                inputs[fmt].encode("utf-8"),
                hashlib.sha256,
            ).digest()
            expected_b64 = base64.b64encode(expected).decode("utf-8")
            if hmac.compare_digest(expected_b64, parsed.signature_b64):
                if not bind_identity and fmt == "legacy":
                    _warn_once(
                        "legacy_sig",
                        "HMAC request signed without X-Tenant/X-User-Id/query binding "
                        "(set HMAC_BIND_IDENTITY=true once clients sign them)",
                    )
                return parsed
    return None


def verify_hmac_request(
    *,
    method: str,
    path: str,
    original_path: str | None,
    body: bytes,
    authorization: str | None,
    timestamp: str | None,
    nonce: str | None,
    keys: Mapping[str, str],
    query: str = "",
    tenant: str | None = None,
    user_id: str | None = None,
) -> bool:
    """Synchronous variant with a process-local nonce store (tests, scripts)."""
    parsed = _verify_signature(
        method=method, path=path, original_path=original_path, body=body,
        authorization=authorization, timestamp=timestamp, nonce=nonce,
        keys=keys, query=query, tenant=tenant, user_id=user_id,
    )
    if parsed is None:
        return False
    # Register the nonce only after a successful verification so that
    # invalid requests cannot burn nonces for legitimate callers.
    return _register_nonce(parsed.key_id, nonce or "", get_settings().hmac_max_skew_seconds * 2)


async def verify_hmac_request_async(**kwargs) -> bool:
    """Request-path variant: nonce is recorded in the shared store (all workers)."""
    parsed = _verify_signature(**kwargs)
    if parsed is None:
        return False
    return await get_nonce_store().register(
        parsed.key_id, kwargs["nonce"] or "", get_settings().hmac_max_skew_seconds * 2
    )
