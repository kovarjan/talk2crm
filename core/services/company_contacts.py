from __future__ import annotations
from typing import List, Dict, Any, Optional

from core.services.vector_search import VectorSearcher
from core.services.company_lookup import find_company_by_name_or_city

def list_contacts_for_account(tenant: str, account_id: str, vector_dir: str = "var/vector") -> List[Dict[str, Any]]:
    """
    Returns ALL contacts whose fields.account_id == account_id from the local metadata.
    No vector search; fast filter over metadata aligned with FAISS index.
    """
    vs = VectorSearcher(tenant, "contacts", vector_dir)
    out: List[Dict[str, Any]] = []
    for rec in vs.meta:  # rec: {"id", "fields": {...}, "relationships": {...}, ...}
        f = rec.get("fields", {})
        if not f:
            continue
        if f.get("account_id") == account_id:
            first = f.get("first_name") or ""
            last  = f.get("last_name") or ""
            display = (first + " " + last).strip()
            out.append({
                "id": rec.get("id"),
                "name": display,
                "email": f.get("email1") or "",
                "phone": f.get("phone_mobile") or f.get("phone_work") or "",
                "account_id": account_id,
                # "raw": rec,
            })

            if len(out) >= 15:
                # Only return names and ids if there are more than 15 contacts
                return [{"id": c["id"], "name": c["name"]} for c in out]
    return out

def list_contacts_for_company_name(query: str, tenant: str, vector_dir: str = "var/vector") -> Dict[str, Any]:
    """
    Resolve company name → account_id with the company lookup, then return contacts for that account.
    Picks the top company hit by score.
    """
    comps = find_company_by_name_or_city(query, tenant=tenant, top_k=1)
    if not comps:
        return {"company": None, "contacts": []}
    acc = comps[0]
    account_id = acc["id"]
    contacts = list_contacts_for_account(tenant, account_id, vector_dir=vector_dir)
    return {"company": acc, "contacts": contacts}
