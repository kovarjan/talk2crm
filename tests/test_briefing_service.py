from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from zoneinfo import ZoneInfo

import pytest

from app.services import briefing_service as service
from database.models import ChatMessage

NOW = datetime(2026, 9, 18, 8, 0, tzinfo=timezone.utc)
MODULES = {b[1].lower() for b in service.BLOCKS}


class CRM:
    def __init__(self, fail=None, rows=None):
        self.calls = []
        self.fail = fail
        self.rows = rows or {}

    async def get_ai_modules(self):
        return [{"name": b[1], "read": True} for b in service.BLOCKS]

    async def execute_module_action(self, *, module, action, data):
        self.calls.append((module, action, data))
        if module == self.fail:
            raise RuntimeError("unavailable")
        return {"records": self.rows.get(module, [])}


def operands(tree):
    if "field" in tree:
        return [tree]
    return [o for child in tree.get("operands", []) for o in operands(child)]


async def collect(crm, **kwargs):
    return await service.collect_briefing(crm_client=crm, tenant_id="t", user_id="u", now=NOW,
                                          timezone_name="Europe/Prague", days=14,
                                          statuses=service.status_sets(), readable=kwargs.get("readable", MODULES))


@pytest.mark.asyncio
async def test_every_section_is_user_scoped_and_read_only():
    crm = CRM()
    result = await collect(crm)
    assert len(crm.calls) == len(result["blocks"]) == 8
    for module, action, payload in crm.calls:
        assert action == "list"
        conditions = operands(payload["filter"])
        assert any(o["field"] == "assigned_user_id" and o["type"] == "eq" and o["value"] == "u" for o in conditions)
        assert payload["limit"] == 101
    assert not result["partial"]
    filters = {m: operands(p["filter"]) for m, _, p in crm.calls}
    assert any(o["field"] == "quote_stage" and o["value"] == "Sent" for o in filters["Quotes"])
    assert any(o["field"] == "datum_splatnosti" and o["value"] == "2026-09-18" and o["type"] == "lessThan" for o in filters["acm_invoices"])
    assert any(o["field"] == "converted" and o["value"] == "0" for o in filters["Leads"])
    assert any(o["field"] == "date_closed" and o["value"] == "2026-10-02" for o in filters["Opportunities"])


@pytest.mark.parametrize("now,start,end", [
    (datetime(2026, 3, 29, 10, tzinfo=timezone.utc), "2026-03-28 23:00:00", "2026-03-29 22:00:00"),
    (datetime(2026, 10, 25, 10, tzinfo=timezone.utc), "2026-10-24 22:00:00", "2026-10-25 23:00:00"),
])
def test_local_day_boundaries_across_daylight_saving(now, start, end):
    filters = operands(service.block_filter("meetings", "Meetings", "date_start", "status", user_id="u", now=now,
                                            zone=ZoneInfo("Europe/Prague"), days=14, statuses=service.status_sets()))
    assert any(o["type"] == "moreThanInclude" and o["value"] == start for o in filters)
    assert any(o["type"] == "lessThan" and o["value"] == end for o in filters)


@pytest.mark.asyncio
async def test_failure_is_partial_denied_modules_not_queried_and_links_are_encoded():
    crm = CRM(fail="Calls", rows={"Quotes": [
        {"id": "q/#?", "name": "Offer", "assigned_user_id": "u"},
        {"id": "foreign", "name": "Not mine", "assigned_user_id": "another-user"},
    ]})
    result = await collect(crm, readable=MODULES - {"acm_invoices"})
    assert result["partial"]
    assert not any(m == "acm_invoices" for m, _, _ in crm.calls)
    blocks = {b["key"]: b for b in result["blocks"]}
    assert blocks["overdue"]["status"] == "unavailable"
    assert blocks["calls"]["status"] == "error"
    assert [r["name"] for r in blocks["quotes"]["records"]] == ["Offer"]
    assert blocks["quotes"]["records"][0]["url"] == "/#detail/Quotes/q%2F%23%3F"
    assert result["cards"][0]["rows"][0]["link"]["url"] == "/#detail/Quotes/q%2F%23%3F"


@pytest.mark.asyncio
async def test_large_lists_are_explicitly_truncated():
    result = await collect(CRM(rows={"Quotes": [{"id": str(i), "name": str(i)} for i in range(101)]}))
    block = next(b for b in result["blocks"] if b["key"] == "quotes")
    assert block["truncated"] and len(block["records"]) == 100


def test_tenant_overrides_only_status_lists_and_validates_shapes():
    result = service.status_sets({"quotes": ["Tenant open"], "orders": [], "assigned_user_id": ["someone else"]})
    assert result["quotes"] == ["Tenant open"]
    assert result["orders"] == []
    assert "assigned_user_id" not in result
    with pytest.raises(ValueError):
        service.status_sets({"quotes": "Active"})


@pytest.mark.asyncio
async def test_focus_summary_is_four_sentences_and_reports_unavailable_data():
    result = await collect(CRM(fail="Calls"))
    summary = service.focus_summary(result)
    assert summary.count('.') == 4
    assert "nedostupný počet hovorů" in summary


@pytest.mark.asyncio
async def test_llm_cannot_introduce_records_or_urls_and_falls_back():
    result = await collect(CRM(rows={"Quotes": [{"id": "q1", "name": "[bad](https://evil.test)"}]}))
    llm = SimpleNamespace(ainvoke=AsyncMock(return_value=SimpleNamespace(content='[999]')))
    with patch.object(service, "get_chat_llm", return_value=llm):
        message = await service.summarize_briefing(result)
    assert "](/#detail/Quotes/q1)" in message
    assert "[bad](https://evil.test)" not in message
    assert llm.ainvoke.await_count == 1


class DB:
    def __init__(self, messages=None, override=None):
        self.messages = messages or []
        self.override = override
        self.added = []
        self.statements = []
        self.commits = 0

    async def execute(self, stmt, params=None):
        self.statements.append((stmt, params))
        if "ai_skill_tenant_overlay" in str(stmt):
            return SimpleNamespace(scalars=lambda: SimpleNamespace(first=lambda: self.override))
        return SimpleNamespace(scalars=lambda: SimpleNamespace(all=lambda: self.messages))

    def add(self, value):
        self.added.append(value)

    async def flush(self):
        pass

    async def commit(self):
        self.commits += 1


@pytest.mark.asyncio
async def test_generation_is_saved_as_ordinary_chat_and_cache_reopens_same_chat():
    db = DB()
    with patch.object(service, "summarize_briefing", AsyncMock(return_value="Your day")):
        result = await service.get_briefing(db=db, crm_client=CRM(), tenant_id="tenant", user_id="user")
    message = next(item for item in db.added if isinstance(item, ChatMessage))
    assert message.tenant_id == "tenant" and message.user_id == "user"
    assert message.content == "Your day"
    assert message.metadata_json["cards"] == result["cards"]
    assert message.chat_id == result["chat_id"]
    assert not result["cached"]
    cached_db = DB(messages=[message])
    crm = CRM()
    cached = await service.get_briefing(db=cached_db, crm_client=crm, tenant_id="tenant", user_id="user")
    assert cached["cached"] and cached["chat_id"] == result["chat_id"]
    assert not crm.calls and not cached_db.added
    query = next(stmt for stmt, _ in cached_db.statements if "JOIN chats" in str(stmt))
    values = query.compile().params
    assert list(values.values()).count("tenant") == 2
    assert list(values.values()).count("user") == 2
    assert any(isinstance(v, datetime) for v in values.values())  # one-hour TTL


@pytest.mark.asyncio
async def test_refresh_bypasses_cache_and_partial_results_are_not_cached():
    db = DB()
    with patch.object(service, "summarize_briefing", AsyncMock(return_value="Partial")):
        result = await service.get_briefing(db=db, crm_client=CRM(fail="Calls"), tenant_id="t", user_id="u", refresh=True)
    assert result["partial"]
    assert not any("JOIN chats" in str(stmt) for stmt, _ in db.statements)
    message = next(item for item in db.added if isinstance(item, ChatMessage))
    assert message.metadata_json["briefing_cache_key"] is None


def test_cache_varies_with_user_tenant_day_timezone_rules_and_permissions():
    args = dict(tenant_id="t", user_id="u", now=NOW, timezone_name="Europe/Prague", days=14,
                statuses=service.status_sets(), readable=MODULES)
    original = service.cache_key(**args)
    for change in [dict(user_id="other"), dict(tenant_id="other"), dict(now=NOW.replace(day=19)),
                   dict(timezone_name="UTC"), dict(days=7), dict(statuses=service.status_sets({"quotes": []})),
                   dict(readable=MODULES - {"quotes"})]:
        assert service.cache_key(**{**args, **change}) != original
