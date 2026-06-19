import time
import pytest
from app.core.stream_token import create_stream_token, validate_stream_token


KEYS = {"k1": "secret-abc"}


def make_token(**overrides):
    params = dict(
        tenant="test-tenant",
        user_id="user-1",
        user_name="Test User",
        key_id="k1",
        secret="secret-abc",
    )
    params.update(overrides)
    return create_stream_token(**params)


def test_valid_token_round_trips():
    token = make_token()
    payload = validate_stream_token(token, hmac_keys=KEYS)
    assert payload["tenant"] == "test-tenant"
    assert payload["user_id"] == "user-1"
    assert payload["user_name"] == "Test User"


def test_wrong_signature_rejected():
    token = make_token()
    parts = token.rsplit(".", 1)
    bad_sig = parts[1][:-1] + ("x" if parts[1][-1] != "x" else "y")
    bad_token = parts[0] + "." + bad_sig
    with pytest.raises(ValueError, match="Invalid token signature"):
        validate_stream_token(bad_token, hmac_keys=KEYS)


def test_unknown_key_id_rejected():
    token = make_token(key_id="unknown", secret="any")
    with pytest.raises(ValueError, match="Unknown key_id"):
        validate_stream_token(token, hmac_keys=KEYS)


def test_malformed_token_rejected():
    with pytest.raises(ValueError, match="Malformed token"):
        validate_stream_token("notavalidtoken", hmac_keys=KEYS)


def test_expired_token_rejected(monkeypatch):
    token = make_token()
    monkeypatch.setattr(
        "app.core.stream_token.time",
        type("T", (), {"time": staticmethod(lambda: time.time() + 120)})(),
    )
    with pytest.raises(ValueError, match="Token expired"):
        validate_stream_token(token, hmac_keys=KEYS)


def test_replay_rejected():
    token = make_token()
    validate_stream_token(token, hmac_keys=KEYS)  # first use: ok
    with pytest.raises(ValueError, match="nonce already used"):
        validate_stream_token(token, hmac_keys=KEYS)  # replay: rejected
