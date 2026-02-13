from __future__ import annotations

import base64
import hashlib
import hmac
import re
import time
from dataclasses import dataclass

from cryptography.fernet import Fernet, InvalidToken

from app.core.config import get_settings


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

    signing_input = "\n".join(
        [
            method.upper(),
            path,
            timestamp,
            nonce,
            body.decode("utf-8", errors="replace"),
        ]
    )
    expected = hmac.new(
        key.encode("utf-8"),
        signing_input.encode("utf-8"),
        hashlib.sha256,
    ).digest()
    expected_b64 = base64.b64encode(expected).decode("utf-8")
    return hmac.compare_digest(expected_b64, parsed.signature_b64)
