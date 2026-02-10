from core.services.crm_service import CRMService, CrmRequestContext
import core.services.crm_service as crm_service


def test_crm_service_routes_context_to_adapter(monkeypatch):
    captured = {}

    def _fake_execute_direct_command(command, tenant=None, user_id=None, user_name=None):
        captured["command"] = command
        captured["tenant"] = tenant
        captured["user_id"] = user_id
        captured["user_name"] = user_name
        return {"ok": True}

    monkeypatch.setattr(crm_service, "execute_direct_command", _fake_execute_direct_command)

    svc = CRMService()
    ctx = CrmRequestContext(tenant="ai-local", user_id="28", user_name="jkovar")
    out = svc.create_record(ctx, module="Contacts", fields={"first_name": "Jan"})

    assert out == {"ok": True}
    assert captured["tenant"] == "ai-local"
    assert captured["user_id"] == "28"
    assert captured["user_name"] == "jkovar"
    assert captured["command"]["action"] == "create"
    assert captured["command"]["module"] == "Contacts"
