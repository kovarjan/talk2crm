from __future__ import annotations
from typing import List, Dict, Any, Optional

from rapidfuzz.fuzz import token_set_ratio

from core.services.vector_search import VectorSearcher   # <-- your existing class
from core.services.textnorm import normalize_cs, digits_only


def _cmp_string(module: str, row: Dict[str, Any]) -> str:
    if module == "contacts":
        name = f"{row.get('first_name','')} {row.get('last_name','')}".strip()
        return f"{name} {row.get('email1','')} {row.get('phone_mobile','')} {row.get('title','')} {row.get('account_name','')}"
    elif module == "accounts":
        return f"{row.get('name','')} {row.get('industry','')} {row.get('billing_address_city','')} {row.get('billing_address_country','')}"
    # meetings
    return f"{row.get('name','')} {row.get('location','')} {row.get('parent_type','')} {row.get('parent_name','')} {row.get('date_start','')}"

def _bonus(query: str, row: Dict[str, Any], module: str, bias: Optional[Dict[str, Any]]) -> float:
    qn = normalize_cs(query)
    bonus = 0.0

    # Email “exact-ish”
    if "@" in query and row.get("email1"):
        if normalize_cs(query) in normalize_cs(row["email1"]):
            bonus += 0.25

    # Phone digits “exact-ish”
    qdigits = digits_only(query)
    if len(qdigits) >= 7 and row.get("phone_mobile"):
        if qdigits in digits_only(row["phone_mobile"]):
            bonus += 0.20

    # Company bias for contacts
    if bias and bias.get("account_name") and module == "contacts":
        if normalize_cs(row.get("account_name","")) == normalize_cs(bias["account_name"]):
            bonus += 0.10

    # Decision-maker titles
    title = normalize_cs(row.get("title",""))
    if any(k in title for k in ("reditel","jednatel","ceo")):
        bonus += 0.05

    return bonus

def _blend(cosine_like: float, fuzz: float, bonus: float) -> float:
    # Tune weights to taste
    return 0.7 * float(cosine_like) + 0.3 * float(fuzz) + float(bonus)

def _map_result(module: str, row: Dict[str, Any], score: float) -> Dict[str, Any]:
    if module == "contacts":
        return {
            "id": row.get("id"),
            "name": f"{row.get('first_name','')} {row.get('last_name','')}".strip(),
            "email": row.get("email1"),
            "phone": row.get("phone_mobile"),
            "account_id": row.get("account_id"),
            "score": score,
            "raw": row,
        }
    if module == "accounts":
        return {
            "id": row.get("id"),
            "name": row.get("name"),
            "city": row.get("billing_address_city"),
            "score": score,
            "raw": row,
        }
    # meetings
    return {
        "id": row.get("id"),
        "name": row.get("name"),
        "date_start": row.get("date_start"),
        "date_end": row.get("date_end"),
        "status": row.get("status"),
        "location": row.get("location"),
        "parent_type": row.get("parent_type"),
        "parent_id": row.get("parent_id"),
        "score": score,
        "raw": row,
    }

def hybrid_module_search(
    module: str,
    query: str,
    tenant: str,
    top_k: int = 5,
    *,
    bias: Optional[Dict[str, Any]] = None,
    fanout: int = 50,
    overview_only: bool = False,
) -> List[Dict[str, Any]]:
    """
    Hybrid search (dense + lexical) over your existing FAISS index:
      - dense: VectorSearcher(search) returns top N with cosine-like score in '_score'
      - lexical: RapidFuzz token_set_ratio over normalized strings
      - extras: email/phone/company bonuses

    Deduplicates results by record id (keeping the highest-scoring hit per id).
    """
    if not tenant or tenant in ("unknown", "none", "null"):
        raise ValueError("tenant is required")

    if module not in ("contacts", "accounts", "meetings"):
        raise ValueError("module must be contacts|accounts|meetings")

    # Dense candidates
    vs = VectorSearcher(tenant=tenant, module=module)
    # Ask for more (fanout) to let lexical rerank do its job
    dense = vs.search(query, top_k=max(top_k, fanout))

    qn = normalize_cs(query)

    # --- NEW: keep only best result per ID ---
    best_by_id: Dict[str, Dict[str, Any]] = {}

    for rec in dense:
        base_cos = float(rec.get("_score", 0.0))  # from your VectorSearcher
        cmp = _cmp_string(module, rec)
        fuzz = token_set_ratio(qn, normalize_cs(cmp)) / 100.0
        b = _bonus(query, rec, module, bias)
        final = _blend(base_cos, fuzz, b)

        rec2 = _map_result(module, rec, final)
        rec2["__cosine"] = base_cos
        rec2["__fuzz"] = float(fuzz)
        rec2["__bonus"] = float(b)

        rid = rec2.get("id")
        if rid is None:
            # if something weird happens and id is missing, just skip/append
            continue

        # Keep only the highest-scoring hit per id
        prev = best_by_id.get(rid)
        if prev is None or rec2["score"] > prev["score"]:
            best_by_id[rid] = rec2

    # Turn dict -> list and sort by final score
    rescored = list(best_by_id.values())
    rescored.sort(key=lambda r: r["score"], reverse=True)
    results = rescored[:top_k]

    if overview_only:
        print("[hybrid_search] overview_only mode: returning id and name only")
        print(results)

        def _get_name(r):
            # Try top-level 'name'
            name = r.get("name")
            if name:
                return name
            # Try raw['fields']['name']
            raw = r.get("raw", {})
            fields = raw.get("fields", {})
            name = fields.get("name")
            if name:
                return name
            # Try raw['name']
            name = raw.get("name")
            if name:
                return name
            # Try contacts fallback
            if module == "contacts":
                first = fields.get("first_name") or raw.get("first_name", "")
                last = fields.get("last_name") or raw.get("last_name", "")
                return f"{first} {last}".strip()
            return ""

        return [
            {
                "id": r["id"],
                "name": _get_name(r),
                "score": r["score"],
            }
            for r in results
        ]

    return results

# Convenience wrappers per module
def search_contacts(query: str, tenant: str, top_k: int = 5, bias_account_name: Optional[str] = None, overview_only: bool = False) -> List[Dict[str, Any]]:
    bias = {"account_name": bias_account_name} if bias_account_name else None
    return hybrid_module_search("contacts", query, tenant, top_k, bias=bias, overview_only=overview_only)

def search_accounts(query: str, tenant: str, top_k: int = 5, overview_only: bool = False) -> List[Dict[str, Any]]:
    return hybrid_module_search("accounts", query, tenant, top_k, overview_only=overview_only)

def search_meetings(query: str, tenant: str, top_k: int = 5, overview_only: bool = False) -> List[Dict[str, Any]]:
    return hybrid_module_search("meetings", query, tenant, top_k, overview_only=overview_only)
