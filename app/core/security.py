from __future__ import annotations

import base64
import hashlib
import hmac
import re
import threading
import time
from dataclasses import dataclass

from cryptography.fernet import Fernet, InvalidToken

from app.core.config import get_settings


# Replay protection: a (key_id, nonce) pair is accepted only once within the
# timestamp skew window. In-process store — sufficient for single-process
# deployments; replace with a shared store (e.g. Redis) when running multiple
# workers behind one endpoint.
_SEEN_NONCES: dict[tuple[str, str], float] = {}
_NONCE_LOCK = threading.Lock()
_NONCE_PRUNE_THRESHOLD = 1024


def _register_nonce(key_id: str, nonce: str, ttl_seconds: float) -> bool:
    """Remember the nonce; returns False when it was already used (replay)."""
    now = time.time()
    with _NONCE_LOCK:
        if len(_SEEN_NONCES) > _NONCE_PRUNE_THRESHOLD:
            for key in [k for k, expiry in _SEEN_NONCES.items() if expiry <= now]:
                del _SEEN_NONCES[key]
        key = (key_id, nonce)
        if _SEEN_NONCES.get(key, 0.0) > now:
            return False
        _SEEN_NONCES[key] = now + ttl_seconds
        return True


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


def verify_hmac_request(
    *,
    method: str,
    path: str,
    original_path: str | None,
    body: bytes,
    authorization: str | None,
    timestamp: str | None,
    nonce: str | None,
) -> bool:
    settings = get_settings()
    parsed = parse_hmac_authorization(authorization or "")
    if not parsed:
        return False
    if not timestamp or not nonce:
        return False

    try:
        ts = int(timestamp)
    except ValueError:
        return False

    now = int(time.time())
    if abs(now - ts) > settings.hmac_max_skew_seconds:
        return False

    key = settings.hmac_keys.get(parsed.key_id)
    if not key:
        return False

    body_text = body.decode("utf-8", errors="replace")
    candidate_paths: list[str] = []
    for candidate in [path, original_path]:
        value = (candidate or "").strip()
        if value and value not in candidate_paths:
            candidate_paths.append(value)

    # Some reverse proxies rewrite /v1/... to /... before forwarding.
    # Try the prefixed path as a second candidate; if neither matches we
    # return False — we do NOT fall through with a partial match.
    if path and not path.startswith("/v1/"):
        prefixed = f"/v1{path}"
        if prefixed not in candidate_paths:
            candidate_paths.append(prefixed)

    for candidate_path in candidate_paths:
        signing_input = "\n".join(
            [
                method.upper(),
                candidate_path,
                timestamp,
                nonce,
                body_text,
            ]
        )
        expected = hmac.new(
            key.encode("utf-8"),
            signing_input.encode("utf-8"),
            hashlib.sha256,
        ).digest()
        expected_b64 = base64.b64encode(expected).decode("utf-8")
        if hmac.compare_digest(expected_b64, parsed.signature_b64):
            # Register the nonce only after a successful verification so that
            # invalid requests cannot burn nonces for legitimate callers.
            return _register_nonce(
                parsed.key_id, nonce, settings.hmac_max_skew_seconds * 2
            )
    return False
