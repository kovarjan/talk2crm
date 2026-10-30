# Copyright 2026 Jan Kovář
# Copyright 2026 Acmark s.r.o.
# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services import tenant_llm
from app.services.tenant_llm import build_llm_config, default_llm_config, resolve_llm_config

SETTINGS = SimpleNamespace(
    llm_base_url="http://system/v1",
    llm_api_key="system-key",
    llm_model="sys-model",
    llm_title_model="sys-title",
    tenant_cache_ttl_seconds=60,
)


@pytest.fixture(autouse=True)
def _clean_cache():
    tenant_llm.invalidate_llm_config_cache()
    yield
    tenant_llm.invalidate_llm_config_cache()


def test_nothing_set_uses_system_default():
    config = build_llm_config(SETTINGS, base_url=" ", api_key=None, model="", title_model=None)
    assert config == default_llm_config(SETTINGS)
    assert not config.custom_endpoint


def test_own_endpoint_never_gets_the_system_key():
    config = build_llm_config(SETTINGS, base_url="http://litellm:4000/v1", model="gemma4")
    assert config.base_url == "http://litellm:4000/v1"
    assert config.api_key == "EMPTY"
    assert config.api_key != SETTINGS.llm_api_key
    assert config.model == "gemma4"
    assert config.title_model == "gemma4"
    assert config.custom_endpoint


def test_own_endpoint_with_key_and_title_model():
    config = build_llm_config(
        SETTINGS, base_url="https://api.openai.com/v1", api_key="sk-x", model="gpt-x", title_model="gpt-mini"
    )
    assert (config.api_key, config.model, config.title_model) == ("sk-x", "gpt-x", "gpt-mini")


def test_url_without_model_uses_system_model_name():
    config = build_llm_config(SETTINGS, base_url="http://other/v1")
    assert config.model == "sys-model"


def test_key_without_url_is_ignored():
    config = build_llm_config(SETTINGS, api_key="orphan-key")
    assert config.api_key == "system-key"
    assert config.base_url == "http://system/v1"


def test_model_only_overrides_model_on_system_endpoint():
    config = build_llm_config(SETTINGS, model="other-model")
    assert (config.base_url, config.api_key) == ("http://system/v1", "system-key")
    assert config.model == "other-model"
    assert config.title_model == "sys-title"


def _session(row):
    result = MagicMock()
    result.one_or_none.return_value = row
    db = MagicMock()
    db.execute = AsyncMock(return_value=result)
    return db


@pytest.mark.asyncio
async def test_resolve_reads_tenant_row_and_caches():
    db = _session(("http://litellm:4000/v1", "sk-t", "qwen3", None, 1.0))
    with patch.object(tenant_llm, "get_settings", return_value=SETTINGS):
        first = await resolve_llm_config("t1", db=db)
        second = await resolve_llm_config("t1", db=db)
    assert first is second
    assert first.base_url == "http://litellm:4000/v1" and first.api_key == "sk-t"
    assert first.temperature == 1.0
    assert db.execute.await_count == 1


@pytest.mark.asyncio
async def test_resolve_without_tenant_or_row_is_system_default_without_db():
    with patch.object(tenant_llm, "get_settings", return_value=SETTINGS):
        assert await resolve_llm_config(None) == default_llm_config(SETTINGS)
        assert await resolve_llm_config("missing", db=_session(None)) == default_llm_config(SETTINGS)


def test_temperature_override_applies_on_both_endpoint_kinds():
    own = build_llm_config(SETTINGS, base_url="http://litellm:4000/v1", temperature=0.0)
    system = build_llm_config(SETTINGS, model="other-model", temperature=0.7)
    assert own.temp(0.3) == 0.0 and system.temp(0.3) == 0.7


def test_no_temperature_override_keeps_each_calls_default():
    config = build_llm_config(SETTINGS, base_url="http://litellm:4000/v1")
    assert config.temperature is None
    assert config.temp(0.0) == 0.0 and config.temp(0.1) == 0.1
