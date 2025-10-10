# core/services/company_lookup.py
from typing import List, Dict, Any
from core.services.vector_search import VectorSearcher

class CompanyLookupService:
    def __init__(self, tenant: str = "ai-local", vector_dir: str = "var/vector"):
        self.searcher = VectorSearcher(tenant, "accounts", vector_dir)

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        hits = self.searcher.search(query, top_k=top_k)
        results: List[Dict[str, Any]] = []
        for h in hits:
            # expected metadata shape from your ingestor/embedder:
            # { "id": "...", "fields": {...}, "relationships": {...}? , "deleted": false, ... }
            fields = h.get("fields", {})
            name  = fields.get("name") or h.get("name") or ""
            city  = fields.get("billing_address_city") or ""
            results.append({
                "id": h.get("id"),
                "name": name,
                "city": city,
                "score": h.get("_score", 0.0),
                "raw": h,  # keep full payload for downstream use
            })
        return results

# convenience function
def find_company_by_name_or_city(query: str, tenant: str = "ai-local", top_k: int = 5):
    return CompanyLookupService(tenant).search(query, top_k=top_k)
