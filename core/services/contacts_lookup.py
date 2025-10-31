# core/services/contacts_lookup.py
from typing import List, Dict, Any
from core.services.vector_search import VectorSearcher

class ContactsLookupService:
    def __init__(self, tenant: str = "ai-local", vector_dir: str = "var/vector"):
        self.searcher = VectorSearcher(tenant, "contacts", vector_dir)

    # def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    #     hits = self.searcher.search(query, top_k=top_k)
    #     results: List[Dict[str, Any]] = []
    #     for h in hits:
    #         f = h.get("fields", {})
    #         first = f.get("first_name") or ""
    #         last  = f.get("last_name") or ""
    #         display = f"{first} {last}".strip()
    #         # if your metadata includes account name (either via enrichment or expand), surface it
    #         acct_name = f.get("account_name") or ""
    #         if acct_name:
    #             display = f"{display}, {acct_name}" if display else acct_name
    #         results.append({
    #             "id": h.get("id"),
    #             "name": display,
    #             "email": f.get("email1") or "",
    #             "phone": f.get("phone_mobile") or f.get("phone_work") or "",
    #             "account_id": f.get("account_id") or h.get("relationships", {}).get("account"),
    #             "score": h.get("_score", 0.0),
    #             "raw": h,
    #         })
    #     return results
    
    def search(self, query: str, top_k: int = 5):
        overfetch = max(top_k * 5, 25)
        hits = self.searcher.search(query, top_k=overfetch)

        by_id = {}
        for h in hits:
            cid = h.get("id")
            if not cid:
                continue
            prev = by_id.get(cid)
            if prev is None or float(h.get("_score", 0.0)) > float(prev.get("_score", 0.0)):
                by_id[cid] = h

        # optional: diacritic-insensitive lexical boost on full_name/email
        def norm(s): 
            import unicodedata
            if not isinstance(s, str): return ""
            return "".join(ch for ch in unicodedata.normalize("NFKD", s) if not unicodedata.combining(ch)).lower().strip()
        nq = norm(query)

        ranked = []
        for h in by_id.values():
            f = h.get("fields", {}) or {}
            name = " ".join([f.get("first_name") or "", f.get("last_name") or ""]).strip() or h.get("full_name") or ""
            email = f.get("email1") or f.get("email") or ""
            base = float(h.get("_score", 0.0))
            boost = 0.0
            nname, nmail = norm(name), norm(email)
            if nq and nq in nname: boost += 0.15
            if nq and nq in nmail: boost += 0.05
            ranked.append((base + boost, h, name))
        ranked.sort(key=lambda x: x[0], reverse=True)

        items = []
        for _, h, name in ranked[:top_k]:
            f = h.get("fields", {}) or {}
            rel = h.get("relationships", {}) or {}
            # normalize relationships -> accounts summary (count+sample)
            acc_ids = []
            if isinstance(rel.get("accounts"), list):
                acc_ids.extend(rel["accounts"])
            if isinstance(rel.get("accounts"), dict):
                acc_ids.extend(rel["accounts"].get("ids") or [])
                acc_ids.extend(rel["accounts"].get("sample") or [])
            if isinstance(rel.get("account"), str):
                acc_ids.append(rel["account"])
            if f.get("account_id"):
                acc_ids.append(f["account_id"])
            # de-dup
            acc_ids = list(dict.fromkeys(a for a in acc_ids if a))

            items.append({
                "id": h.get("id"),
                "name": name or h.get("name") or "",
                "email": f.get("email1") or f.get("email") or "",
                "phone": f.get("phone_mobile") or "",
                "account_id": f.get("account_id") or (rel.get("account") if isinstance(rel.get("account"), str) else None),
                "score": h.get("_score", 0.0),
                "raw": {
                    "id": h.get("id"),
                    "_index": h.get("_index"),
                    "_score": h.get("_score"),
                    "date_modified": h.get("date_modified"),
                    "deleted": h.get("deleted", False),
                    "fields": f,
                    "relationships": {
                        "accounts": {
                            "count": len(acc_ids),
                            "sample": acc_ids[:10],
                        }
                    },
                }
            })

        return items


def find_contact_by_query(query: str, tenant: str = "ai-local", top_k: int = 5):
    return ContactsLookupService(tenant).search(query, top_k=top_k)
