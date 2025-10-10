# core/services/contacts_lookup.py
from typing import List, Dict, Any
from core.services.vector_search import VectorSearcher

class ContactsLookupService:
    def __init__(self, tenant: str = "ai-local", vector_dir: str = "var/vector"):
        self.searcher = VectorSearcher(tenant, "contacts", vector_dir)

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        hits = self.searcher.search(query, top_k=top_k)
        results: List[Dict[str, Any]] = []
        for h in hits:
            f = h.get("fields", {})
            first = f.get("first_name") or ""
            last  = f.get("last_name") or ""
            display = f"{first} {last}".strip()
            # if your metadata includes account name (either via enrichment or expand), surface it
            acct_name = f.get("account_name") or ""
            if acct_name:
                display = f"{display}, {acct_name}" if display else acct_name
            results.append({
                "id": h.get("id"),
                "name": display,
                "email": f.get("email1") or "",
                "phone": f.get("phone_mobile") or f.get("phone_work") or "",
                "account_id": f.get("account_id") or h.get("relationships", {}).get("account"),
                "score": h.get("_score", 0.0),
                "raw": h,
            })
        return results

def find_contact_by_query(query: str, tenant: str = "ai-local", top_k: int = 5):
    return ContactsLookupService(tenant).search(query, top_k=top_k)
