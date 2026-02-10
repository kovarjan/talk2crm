import core.adapters.crm_direct as crm_direct
from core.clients.client_loader import RestClientConfig


def _sugar_cfg() -> RestClientConfig:
    return RestClientConfig(
        name="ai-local",
        rest_url="http://localhost:2000/public",
        rest_hmac_key_id="k",
        rest_hmac_secret="s",
        crm_direct_backend="sugar_v4_1",
        sugar_rest_url="https://localhost:2000/service/v4_1/rest.php",
        sugar_auth_mode="session",
        sugar_session_id="sid-123",
    )


def test_sugar_filter_to_query_eq_and_cont():
    eq = crm_direct._sugar_filter_to_query({"operator": "eq", "operands": ["last_name", "Novak"]})
    assert "last_name = 'Novak'" == eq

    cont = crm_direct._sugar_filter_to_query({"operator": "cont", "operands": ["name", "Acme"]})
    assert "name LIKE '%Acme%'" == cont


def test_execute_direct_command_uses_sugar_get_entry_list(monkeypatch):
    captured = {}

    monkeypatch.setattr(crm_direct, "get_rest_client_config", lambda tenant: _sugar_cfg())

    def _fake_sugar_request(rest_url, method, rest_data, timeout=20):
        captured["rest_url"] = rest_url
        captured["method"] = method
        captured["rest_data"] = rest_data
        return {
            "entry_list": [
                {
                    "id": "rec-1",
                    "module_name": "Contacts",
                    "name_value_list": [
                        {"name": "first_name", "value": "Jan"},
                        {"name": "last_name", "value": "Novak"},
                    ],
                }
            ],
            "next_offset": -1,
        }

    monkeypatch.setattr(crm_direct, "_sugar_request", _fake_sugar_request)

    result = crm_direct.execute_direct_command(
        {
            "action": "list",
            "module": "contacts",
            "parameters": {
                "filter": {"operator": "eq", "operands": ["last_name", "Novak"]},
                "limit": 5,
                "offset": 0,
            },
        },
        tenant="ai-local",
        user_id="28",
    )

    assert captured["method"] == "get_entry_list"
    assert captured["rest_data"]["session"] == "sid-123"
    assert captured["rest_data"]["module_name"] == "Contacts"
    assert "last_name = 'Novak'" in captured["rest_data"]["query"]
    assert result["module"] == "Contacts"
    assert result["records"][0]["first_name"] == "Jan"


def test_get_available_modules_uses_sugar_backend(monkeypatch):
    captured = {}

    monkeypatch.setattr(crm_direct, "get_rest_client_config", lambda tenant: _sugar_cfg())

    def _fake_sugar_request(rest_url, method, rest_data, timeout=20):
        captured["method"] = method
        captured["rest_data"] = rest_data
        return {"modules": ["Accounts", "Contacts"]}

    monkeypatch.setattr(crm_direct, "_sugar_request", _fake_sugar_request)

    out = crm_direct.get_available_modules(
        tenant="ai-local",
        user_id="28",
        user_name="jkovar",
    )

    assert captured["method"] == "get_available_modules"
    assert captured["rest_data"]["session"] == "sid-123"
    assert out["modules"] == ["Accounts", "Contacts"]
