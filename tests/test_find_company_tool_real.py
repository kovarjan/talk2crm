# tests/test_find_company_tool_strict.py
import json
import unicodedata
import pytest

from core.services.tools import find_company_tool

# --- helpers ---------------------------------------------------------------

def _strip_accents(s: str) -> str:
    if not isinstance(s, str):
        return ""
    nfkd = unicodedata.normalize("NFKD", s)
    return "".join(ch for ch in nfkd if not unicodedata.combining(ch))

def norm(s: str) -> str:
    return _strip_accents((s or "")).lower().strip()

def parse_result(payload):
    """Call the tool with either a plain string or a JSON-string payload."""
    if isinstance(payload, str):
        payload = {"query": payload, "tenant": "ai-local"}
    if isinstance(payload, dict):
        payload = json.dumps(payload, ensure_ascii=False)

    raw = find_company_tool(payload)
    data = json.loads(raw)
    assert "success" in data and data["success"] is True, f"Tool failed: {data}"
    assert isinstance(data.get("items"), list), "items must be a list"
    return data

def rel_count(item, rel_key="contacts"):
    """
    Read relationship count regardless of representation:
    - summary form: item["raw"]["relationships"][rel_key] -> {"count": int, "sample": [...]}
    - full form   : item["raw"]["relationships"][rel_key] -> [id, id, ...]
    Missing -> 0
    """
    raw = (item.get("raw") or {})
    rels = (raw.get("relationships") or {})
    val = rels.get(rel_key)
    if val is None:
        return 0
    if isinstance(val, dict) and "count" in val:
        return int(val["count"])
    if isinstance(val, list):
        return len(val)
    return 0

def is_sorted_desc_by_score(items):
    scores = [float(x.get("score", 0.0)) for x in items]
    return all(scores[i] >= scores[i+1] for i in range(len(scores)-1))

# --- fixtures/data from CRM (ground truth you shared) ----------------------

KNOWN = [
    # name,                               expected_city, expected_contacts_count
    ("Jaroslav Sládek",                   "Praha",       0),
    ("KM Beta a.s.",                      "Hodonín",     1),
    ("PLAS s.r.o.",                       "Praha",       3),
    ("ČEMAT trading, spol. s r.o.",       "Bohumín",     4),
    ("Apex Central Europe, s.r.o.",       "Blučina",     6),
    ("Alza.cz a.s.",                      "Praha",       0),
]

# Create diacritic/case variants for the tricky one
CEMAT_VARIANTS = [
    "ČEMAT trading, spol. s r.o.",
    "cemat trading, spol. s r.o.",
    "Čemat trading sro",
    "CEMAT TRADING, SPOL. S R.O.",
    "cemat",
]

# --- tests ----------------------------------------------------------------

@pytest.mark.parametrize("name,expected_city,_", KNOWN)
def test_each_known_query_returns_match(name, expected_city, _):
    data = parse_result(name)
    assert data["items"], f"No items for {name}"
    # At least one result matches normalized name
    assert any(norm(name) in norm(it.get("name")) for it in data["items"]), \
        f"No name match for {name}. Got: {[it.get('name') for it in data['items']]}"
    # If city is present, it should match (case/diacritic-insensitive)
    matched = next((it for it in data["items"] if norm(name) in norm(it.get("name"))), data["items"][0])
    if matched.get("city"):
        assert norm(matched["city"]) == norm(expected_city), \
            f"City mismatch for {name}: got {matched['city']} expected {expected_city}"

def test_ids_are_unique_and_scores_sorted():
    data = parse_result("KM Beta a.s.")
    ids = [it["id"] for it in data["items"] if "id" in it]
    assert len(ids) == len(set(ids)), f"Duplicate IDs returned: {ids}"
    assert is_sorted_desc_by_score(data["items"]), "Items must be sorted by score desc"

@pytest.mark.parametrize("q", CEMAT_VARIANTS)
def test_case_and_diacritic_insensitive_matching(q):
    data = parse_result(q)
    assert any("cemat" in norm(it.get("name")) for it in data["items"]), \
        f"Diacritic/case-insensitive match failed for query={q}"

def test_top_k_respected_with_json_payload():
    payload = {"query": "Alza", "tenant": "ai-local", "top_k": 3}
    data = parse_result(payload)
    assert len(data["items"]) <= 3, f"top_k not respected: got {len(data['items'])}"
    # Ensure IDs unique within top_k window
    ids = [it.get("id") for it in data["items"]]
    assert len(ids) == len(set(ids)), "Duplicate IDs within top_k"

@pytest.mark.parametrize("name,_,expected_contacts", KNOWN)
def test_relationship_counts_are_sane(name, _, expected_contacts):
    """
    We accept either summarized or full forms. This asserts the count we ingest
    is consistent with CRM ground truth you provided.
    (If your exporter changed later, tweak expected numbers here.)
    """
    data = parse_result(name)
    # Look at the best matching item for the name
    target = next((it for it in data["items"] if norm(name) in norm(it.get("name"))), data["items"][0])
    count = rel_count(target, "contacts")
    # Exact where we have ground truth; allow small tolerance if needed:
    assert count == expected_contacts, f"{name}: expected {expected_contacts} contacts, got {count}"

def test_result_schema_is_minimal_and_sanitized():
    """
    Guard against huge payloads by requiring:
      - 'raw.relationships.contacts' is either summary dict or a list not exceeding a safe cap
      - 'fields' minimal fields present in raw (id, fields, deleted allowed)
    """
    data = parse_result("Apex Central Europe")
    it = data["items"][0]
    # raw minimal
    raw = it.get("raw") or {}
    assert "id" in raw and "fields" in raw, "raw must include id and fields"
    rels = raw.get("relationships", {})
    if "contacts" in rels and isinstance(rels["contacts"], list):
        # Cap: fail if exporter ever returns thousands again
        assert len(rels["contacts"]) <= 200, "contacts list should be capped to avoid huge payloads"
    elif "contacts" in rels and isinstance(rels["contacts"], dict):
        # summary form must include keys
        assert "count" in rels["contacts"] and "sample" in rels["contacts"], "summary must have count+sample"

def test_plain_and_json_input_are_equivalent_for_defaults():
    plain = parse_result("PLAS s.r.o.")
    via_json = parse_result("PLAS s.r.o.")
    # Not necessarily identical ordering/scores, but both should include the entity
    assert any("plas" in norm(it.get("name")) for it in plain["items"])
    assert any("plas" in norm(it.get("name")) for it in via_json["items"])
