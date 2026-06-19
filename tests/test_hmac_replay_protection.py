from __future__ import annotations

import base64
import hashlib
import hmac
import sys
import time
import uuid
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import app.core.security as security


KEY_ID = "test-kid"
SECRET = "test-secret-1234567890"


def _fake_settings():
    return SimpleNamespace(
        hmac_keys={KEY_ID: SECRET},
        hmac_max_skew_seconds=300,
    )


def _signed_request(*, nonce: str, timestamp: str | None = None, secret: str = SECRET) -> dict:
    method = "POST"
    path = "/process-input/"
    body = b'{"input_text":"test"}'
    ts = timestamp or str(int(time.time()))
    signing_input = "\n".join([method, path, ts, nonce, body.decode("utf-8")])
    signature = base64.b64encode(
        hmac.new(secret.encode("utf-8"), signing_input.encode("utf-8"), hashlib.sha256).digest()
    ).decode("utf-8")
    return {
        "method": method,
        "path": path,
        "original_path": None,
        "body": body,
        "authorization": f"HMAC keyId={KEY_ID}, signature={signature}",
        "timestamp": ts,
        "nonce": nonce,
    }


def test_valid_request_passes(monkeypatch) -> None:
    monkeypatch.setattr(security, "get_settings", _fake_settings)
    assert security.verify_hmac_request(**_signed_request(nonce=uuid.uuid4().hex)) is True


def test_replayed_nonce_is_rejected(monkeypatch) -> None:
    monkeypatch.setattr(security, "get_settings", _fake_settings)
    request = _signed_request(nonce=uuid.uuid4().hex)
    assert security.verify_hmac_request(**request) is True
    assert security.verify_hmac_request(**request) is False


def test_fresh_nonce_passes_after_replay_rejection(monkeypatch) -> None:
    monkeypatch.setattr(security, "get_settings", _fake_settings)
    assert security.verify_hmac_request(**_signed_request(nonce=uuid.uuid4().hex)) is True
    assert security.verify_hmac_request(**_signed_request(nonce=uuid.uuid4().hex)) is True


def test_invalid_signature_does_not_burn_the_nonce(monkeypatch) -> None:
    monkeypatch.setattr(security, "get_settings", _fake_settings)
    nonce = uuid.uuid4().hex
    bad = _signed_request(nonce=nonce, secret="wrong-secret")
    assert security.verify_hmac_request(**bad) is False
    # The legitimate caller can still use the same nonce afterwards.
    assert security.verify_hmac_request(**_signed_request(nonce=nonce)) is True


def test_stale_timestamp_rejected(monkeypatch) -> None:
    monkeypatch.setattr(security, "get_settings", _fake_settings)
    stale = str(int(time.time()) - 3600)
    request = _signed_request(nonce=uuid.uuid4().hex, timestamp=stale)
    assert security.verify_hmac_request(**request) is False
