# core/services/company_lookup.py
from typing import List, Dict, Any, Tuple
import unicodedata

from core.services.vector_search import VectorSearcher

# --- Normalization helpers (case + diacritics) ---

def _strip_accents(s: str) -> str:
    if not isinstance(s, str):
        return ""
    nfkd = unicodedata.normalize("NFKD", s)
    return "".join(ch for ch in nfkd if not unicodedata.combining(ch))

def _norm(s: str) -> str:
    return _strip_accents(s).lower().strip()

# Prefer non-empty a - otherwise fall back to b
def _merge_non_empty(a: Any, b: Any) -> Any:
    if a not in (None, "", [], {}):
        return a
    return b

def _summarize_relationships(rels: Dict[str, List[str]], sample_size: int = 10) -> Dict[str, Dict[str, Any]]:
    """
    Convert large relationship arrays into { type: {count: int, sample: [ids...] } }.
    This prevents massive payloads and bogus '6000 contacts' UI issues.
    """
    if not isinstance(rels, dict):
        return {}

    out: Dict[str, Dict[str, Any]] = {}
    for rtype, ids in rels.items():
        if not isinstance(ids, list):
            continue
        count = len(ids)
        sample = ids[:sample_size]
        out[rtype] = {"count": count, "sample": sample}
    return out

class CompanyLookupService:
    def __init__(self, tenant: str = "ai-local", vector_dir: str = "var/vector"):
        self.searcher = VectorSearcher(tenant, "accounts", vector_dir)

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        # Over-fetch to allow dedup + re-rank
        overfetch = max(top_k * 5, 25)
        hits = self.searcher.search(query, top_k=overfetch)

        # Aggregate by company ID; keep best score and merge fields
        by_id: Dict[str, Dict[str, Any]] = {}
        for h in hits:
            cid = h.get("id")
            if not cid:
                # Skip malformed rows
                continue

            prev = by_id.get(cid)
            if prev is None:
                by_id[cid] = h
            else:
                # Keep the one with the higher score
                prev_score = prev.get("_score", 0.0)
                cur_score = h.get("_score", 0.0)
                best = h if cur_score > prev_score else prev

                # Merge a few useful fields non-destructively
                merged = dict(best)
                merged_fields = dict(merged.get("fields", {}))
                prev_fields = prev.get("fields", {})
                cur_fields = h.get("fields", {})
                merged_fields["name"] = _merge_non_empty(
                    prev_fields.get("name"), cur_fields.get("name")
                )
                merged_fields["billing_address_city"] = _merge_non_empty(
                    prev_fields.get("billing_address_city"), cur_fields.get("billing_address_city")
                )
                merged["fields"] = merged_fields

                # Prefer non-empty date_modified/deleted if needed (optional)
                merged["date_modified"] = _merge_non_empty(prev.get("date_modified"), h.get("date_modified"))
                merged["deleted"] = _merge_non_empty(prev.get("deleted"), h.get("deleted"))

                by_id[cid] = merged

        # Re-rank with a simple lexical boost (case/diacritic insensitive)
        nq = _norm(query)
        ranked: List[Tuple[float, Dict[str, Any]]] = []
        for h in by_id.values():
            fields = h.get("fields", {}) or {}
            name = fields.get("name") or h.get("name") or ""
            city = fields.get("billing_address_city") or ""
            base = float(h.get("_score", 0.0))

            # Lexical boost if query appears in name/city after normalization
            boost = 0.0
            n_name = _norm(name)
            n_city = _norm(city)
            if nq and (nq in n_name):
                boost += 0.15
            if nq and (nq in n_city):
                boost += 0.05

            ranked.append((base + boost, h))

        ranked.sort(key=lambda x: x[0], reverse=True)

        # Build final results with sanitized relationships
        results: List[Dict[str, Any]] = []
        for _, h in ranked[:top_k]:
            fields = h.get("fields", {}) or {}
            rels = h.get("relationships", {}) or {}
            rels_summary = _summarize_relationships(rels, sample_size=10)

            item = {
                "id": h.get("id"),
                "name": fields.get("name") or h.get("name") or "",
                "city": fields.get("billing_address_city") or "",
                "score": h.get("_score", 0.0),  # original score (not including local boost)
                # Provide a smaller, safe 'raw' for downstream usage:
                "raw": {
                    "id": h.get("id"),
                    "fields": fields,
                    "relationships": rels_summary,  # summarized, not full arrays
                    "deleted": h.get("deleted", False),
                    "date_modified": h.get("date_modified"),
                    "_index": h.get("_index"),
                    "_score": h.get("_score"),
                },
            }
            results.append(item)

        return results

# convenience function
def find_company_by_name_or_city(query: str, tenant: str = "ai-local", top_k: int = 5):
    return CompanyLookupService(tenant).search(query, top_k=top_k)
