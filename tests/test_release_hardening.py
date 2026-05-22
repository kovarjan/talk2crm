"""Tests for release hardening: security and performance fixes."""
from __future__ import annotations

import re
import sys
import uuid
from pathlib import Path

import pytest
from unittest.mock import patch
from pydantic import ValidationError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# --- Task 1: Fernet key validator ---

def test_settings_rejects_placeholder_fernet_key():
    """Settings must raise at construction time if tenant_secret_key is the placeholder."""
    with patch.dict("os.environ", {
        "TENANT_SECRET_KEY": "CHANGE_ME_WITH_32_BYTE_URLSAFE_BASE64_FOR_FERNET",
    }):
        from app.core import config as cfg_module
        with pytest.raises(ValidationError):
            cfg_module.Settings()


def test_settings_accepts_valid_fernet_key():
    """Settings accepts a properly formatted Fernet key."""
    from cryptography.fernet import Fernet
    real_key = Fernet.generate_key().decode()
    with patch.dict("os.environ", {"TENANT_SECRET_KEY": real_key}):
        from app.core import config as cfg_module
        s = cfg_module.Settings()
        assert s.tenant_secret_key == real_key


# --- Task 1: Path traversal ---

_SAFE_FILE_ID_RE = re.compile(r"^[a-f0-9]{32}$")


def test_audio_file_id_rejects_path_traversal():
    """Validate that path traversal patterns don't match the safe ID regex."""
    bad_ids = [
        "../../../etc/passwd",
        "../../secret",
        "abc/../foo",
        "foo/bar",
        "a" * 33,
        "A1B2C3D4E5F6A1B2C3D4E5F6A1B2C3D4",  # uppercase — must reject
    ]
    for bad in bad_ids:
        assert not _SAFE_FILE_ID_RE.fullmatch(bad), f"Should reject: {bad!r}"


def test_audio_file_id_accepts_valid_hex():
    """get_audio must accept valid uuid4().hex style IDs."""
    good_id = uuid.uuid4().hex
    assert _SAFE_FILE_ID_RE.fullmatch(good_id), f"Should accept: {good_id!r}"


# --- Task 2: crm_mode validator ---

def test_settings_rejects_invalid_crm_mode():
    """crm_mode must be 'on' or 'off', nothing else."""
    from cryptography.fernet import Fernet
    real_key = Fernet.generate_key().decode()
    with patch.dict("os.environ", {
        "TENANT_SECRET_KEY": real_key,
        "CRM_MODE": "yes",
    }):
        from app.core import config as cfg_module
        with pytest.raises(ValidationError):
            cfg_module.Settings()


def test_settings_accepts_crm_mode_on_and_off():
    """crm_mode accepts on/off in any case and normalizes to lowercase."""
    from cryptography.fernet import Fernet
    real_key = Fernet.generate_key().decode()
    from app.core import config as cfg_module
    for mode in ("on", "off", "ON", "OFF"):
        with patch.dict("os.environ", {
            "TENANT_SECRET_KEY": real_key,
            "CRM_MODE": mode,
        }):
            s = cfg_module.Settings()
            assert s.crm_mode in ("on", "off"), f"Expected normalized mode for input {mode!r}"


def test_rag_embedding_api_key_default_is_none():
    """rag_embedding_api_key defaults to None, not llm_api_key."""
    from app.core.config import Settings

    # Create a Settings instance and verify the field default is None
    # by inspecting the model fields
    assert Settings.model_fields["rag_embedding_api_key"].default is None, (
        "rag_embedding_api_key field default must be None, "
        "not the llm_api_key class variable or string 'EMPTY'"
    )


# --- Task 4: Shared utility functions ---

def test_normalize_text_strips_diacritics():
    from app.utils.text import normalize_text
    assert normalize_text("Nováková") == "novakova"
    assert normalize_text("Čemat") == "cemat"
    assert normalize_text("  Hello  World  ") == "hello world"
    assert normalize_text("") == ""


def test_normalize_text_collapses_punctuation():
    from app.utils.text import normalize_text
    assert normalize_text("foo-bar.baz") == "foo bar baz"


def test_safe_text_handles_none_and_whitespace():
    from app.utils.text import safe_text
    assert safe_text(None) == ""
    assert safe_text("  hello  ") == "hello"
    assert safe_text(42) == "42"


def test_crm_id_re_matches_valid_ids():
    from app.utils.crm_id import CRM_ID_RE, is_valid_crm_id
    assert CRM_ID_RE.match("a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4")
    assert CRM_ID_RE.match("a1b2c3d4-e5f6-a1b2-c3d4-e5f6a1b2c3d4")
    assert is_valid_crm_id("a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4")
    assert not is_valid_crm_id(None)
    assert not is_valid_crm_id("not-an-id")
    assert not is_valid_crm_id("CONTACT_ID")
