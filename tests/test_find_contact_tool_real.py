# tests/test_find_contact_tool.py
import json
import unicodedata
import pytest

from core.services.tools import find_contact_tool

# ----------------- helpers -----------------

def _strip_accents(s: str) -> str:
    if not isinstance(s, str):
        return ""
    nfkd = unicodedata.normalize("NFKD", s)
    return "".join(ch for ch in nfkd if not unicodedata.combining(ch))

def norm(s: str) -> str:
    return _strip_accents((s or "")).lower().strip()

def call_tool(payload):
    if isinstance(payload, str):
        payload = {"query": payload, "tenant": "ai-local"}
    if isinstance(payload, dict):
        payload = json.dumps(payload, ensure_ascii=False)
    raw = find_contact_tool(payload)
    data = json.loads(raw)
    assert "success" in data and data["success"] is True, f"Tool failed: {data}"
    assert isinstance(data.get("items"), list), "items must be a list"
    return data

def get_relationship_ids(item, rel_key="accounts"):
    raw = item.get("raw") or {}
    rels = raw.get("relationships") or {}
    out = []

    # Preferred future schema (plural list/summary)
    v = rels.get(rel_key)
    if isinstance(v, dict):
        out.extend(v.get("sample") or [])
        if isinstance(v.get("ids"), list):
            out.extend(v["ids"])
    elif isinstance(v, list):
        out.extend(v)

    # CURRENT exporter shape: singular scalar id
    if isinstance(rels.get("account"), str):
        out.append(rels["account"])

    # Also accept fields.account_id
    fields = raw.get("fields") or {}
    if isinstance(fields.get("account_id"), str):
        out.append(fields["account_id"])

    # de-dup
    return list(dict.fromkeys(out))


def best_match(items, name_or_id):
    """
    Find the best item by exact id; otherwise by normalized name substring.
    """
    # 1) ID match
    if isinstance(name_or_id, dict) and name_or_id.get("id"):
        for it in items:
            if it.get("id") == name_or_id["id"]:
                return it
    # 2) Name match
    target_name = name_or_id["name"] if isinstance(name_or_id, dict) else name_or_id
    n_target = norm(target_name)
    for it in items:
        if n_target in norm(it.get("name")):
            return it
    return None

def is_sorted_desc_by_score(items):
    scores = [float(x.get("score", 0.0)) for x in items]
    return all(scores[i] >= scores[i+1] for i in range(len(scores)-1))

# ----------------- ground truth fixture -----------------
# Fill with your CRM export. Each entry:
#   - name (required)
#   - id (optional but recommended)
#   - expected_account_id (required): company/account id the contact is linked to
#   - city/email/phone optional (used by some assertions if provided)
CONTACTS = [
    {
        "name": "Lucia Adamcová",
        "id": 'a9ce35ab-8db6-f4f9-b934-61556c9dbce4',
        "expected_account_id": "7d956317-c1ba-0118-1c91-61545f91b878",
        "city": 'Bratislava',
        "email": 'lucia.adamcova@365.bank',
    },
    {
        "name": "Libor Adamec",
        "id": '938c407f-022c-fbe1-54b1-67463fa4ceb8',
        "expected_account_id": "95ced180-9695-4379-a2be-f7b237d73114",
        "city": None,
        "email": 'libor.adamec@gatema.cz',
    },
    {
        "name": "Jiří Adamuška",
        "id": "dc2a6535-b07c-a43e-3e27-6151d9192390",
        "expected_account_id": "51c61c4d-1393-21fd-ba0d-6151d9593d19",
        "city": "Praha 10",
        "email": "jaromir.adamuska@mzp.cz",
    },
    {
        "name": "Stanislava Alexová",
        "id": "ad421a03-db9e-5a25-390d-660e8e405cc0",
        "expected_account_id": "777e792e-2ef9-15b9-2da8-65f995e823d1",
        "city": "Brno",
        "email": "alexova@bclogia.cz",
    },
    # {
    #     "name": "Karel Antonín",
    #     "id": "b8b585b8-2d88-3d7b-e6e8-4df0af1c1b57",
    #     "expected_account_id": "d1205243-d478-c58f-5d5b-4d39e868318f",
    #     "city": "Humpolec",
    #     "email": "antonin@ok-strojservis.cz",
    # },
]

# Diacritic/case variants for resilience
NAME_VARIANTS = {
    "Č/á variants": [
        "Lucia Adamcová",
        "lucia adamcova",
        "LUCIA ADAMCOVÁ",
        # "Lucie Adamcova",
    ],
}

# ----------------- tests -----------------

@pytest.mark.parametrize("contact", CONTACTS)
def test_find_contact_exact_and_relationships(contact):
    """
    Exact contact should be present; and its relationships MUST
    include the expected company/account id.
    """
    data = call_tool(contact["name"])
    items = data["items"]


    assert items, f"No items for {contact['name']}"
    chosen = best_match(items, contact)

    # print("DEBUG chosen item:\n", json.dumps(chosen, ensure_ascii=False, indent=2))

    assert chosen is not None, f"Not found: {contact['name']} (by id or name)"
    # If contact['id'] provided, ensure exact ID match
    if contact.get("id"):
        assert chosen["id"] == contact["id"], f"ID mismatch for {contact['name']}"
    # Relationships: expect account id present
    rel_ids = get_relationship_ids(chosen, "accounts")
    # Support both "accounts" (list/dict) and "account" (single id)
    # print("DEBUG rel_ids:", rel_ids)
    if not rel_ids and "account" in (chosen.get("raw") or {}).get("relationships", {}):
        rel_ids = [(chosen.get("raw") or {}).get("relationships", {}).get("account")]
    assert contact["expected_account_id"] in rel_ids, (
        f"Expected account {contact['expected_account_id']} not present for contact {contact['name']}. "
        f"Got relationships.accounts={rel_ids}"
    )
    # Optional field checks if provided
    if contact.get("email"):
        assert norm(chosen.get("email")) == norm(contact["email"]), f"Email mismatch for {contact['name']}"

    # print("DEBUG item names & rels:", [
    #     {"name": it.get("name"),
    #     "id": it.get("id"),
    #     "rels": (it.get("raw") or {}).get("relationships")}
    #     for it in items
    # ])


@pytest.mark.parametrize("variant_group", NAME_VARIANTS.values())
def test_case_and_diacritic_insensitive_queries(variant_group):
    """
    Any variant of the same name should still find a matching contact.
    """
    base = variant_group[0]
    for q in variant_group:
        data = call_tool(q)
        assert any(norm(base) in norm(it.get("name")) for it in data["items"]), \
            f"Query variant failed: {q}"

def test_ids_unique_and_scores_sorted_for_generic_query():
    data = call_tool("Petr")
    ids = [it.get("id") for it in data["items"] if it.get("id")]
    assert len(ids) == len(set(ids)), "Duplicate IDs returned"
    assert is_sorted_desc_by_score(data["items"]), "Items must be sorted by score desc"

def test_top_k_respected_with_json_payload():
    payload = {"query": "Adam", "tenant": "ai-local", "top_k": 4}
    data = call_tool(payload)
    assert len(data["items"]) <= 4, f"top_k not respected: got {len(data['items'])}"
    # Still unique within the window
    ids = [it.get("id") for it in data["items"] if it.get("id")]
    assert len(ids) == len(set(ids)), "Duplicate IDs within top_k"

def test_schema_sane_and_relationship_payload_not_exploding():
    """
    Guardrails: ensure raw exists and relationships are reasonably sized.
    """
    data = call_tool("Abrah")
    it = data["items"][0]
    raw = it.get("raw") or {}
    assert "id" in raw and "fields" in raw, "raw must include id and fields"
    rels = raw.get("relationships", {})
    # If a full list is returned, cap must be in effect (≤200)
    if "accounts" in rels and isinstance(rels["accounts"], list):
        assert len(rels["accounts"]) <= 200, "accounts list should be capped"
    # If summary is returned, it must have count/sample
    if "accounts" in rels and isinstance(rels["accounts"], dict):
        assert "count" in rels["accounts"] and "sample" in rels["accounts"], "summary must have count+sample"

@pytest.mark.parametrize("contact", CONTACTS)
def test_plain_vs_json_payload_equivalence(contact):
    """
    Plain string vs JSON payload should both find the same person (not necessarily same order/score).
    """
    plain = call_tool(contact["name"])
    via_json = call_tool({"query": contact["name"], "tenant": "ai-local", "top_k": 5})
    def has_tgt(items):
        if contact.get("id"):
            return any(it.get("id") == contact["id"] for it in items)
        return any(norm(contact["name"]) in norm(it.get("name")) for it in items)
    assert has_tgt(plain["items"]), "Target not found with plain query"
    assert has_tgt(via_json["items"]), "Target not found with JSON query"

# def test_contact_tool_karel_antonin_manual():
#     ret = find_contact_tool("Karel Antonín OK Strojservis")
#     print("Manual test output:\n", ret)
#     data = json.loads(ret)
#     assert data.get("success") is True, "Tool failed in manual test"

def test_contact_tool_not_full_name():
    data = call_tool("paní Mimrová z Panasu")
    assert any("mimrova" in norm(it.get("name")) for it in data["items"]), "Expected contact not found"
    assert any("karolina" in norm(it.get("name")) for it in data["items"]), "Expected Karolína Mimrová not found"
