from sentence_transformers import SentenceTransformer
import faiss, numpy as np, pickle, os, json

class Embedder:
    def __init__(self, model_name="all-MiniLM-L6-v2", dim=384, store_dir="var/vector"):
        self.model = SentenceTransformer(model_name)
        self.dim, self.store_dir = dim, store_dir
        os.makedirs(store_dir, exist_ok=True)

    def build_text_accounts(self, rec):
        f = rec["fields"]
        parts = [f.get("name",""), f.get("industry",""), f.get("account_type",""),
                 f.get("phone_office",""), f.get("email1",""),
                 f.get("billing_address_city",""), f.get("billing_address_country",""),
                 f.get("description","")]
        return ", ".join([p for p in parts if p])

    def build_text_contacts(self, rec):
        f = rec["fields"]
        name = " ".join([f.get("first_name",""), f.get("last_name","")]).strip()
        parts = [name, f.get("title",""), f.get("department",""), f.get("email1",""), f.get("phone_mobile",""), f.get("description","")]
        return ", ".join([p for p in parts if p])

    def upsert_batch(self, tenant: str, module: str, records: list):
        ns = f"{tenant}:{module}"
        texts, ids, meta = [], [], []
        for r in records:
            if r.get("deleted"): continue
            if module == "accounts": text = self.build_text_accounts(r)
            elif module == "contacts": text = self.build_text_contacts(r)
            else: text = self._generic_text(r)
            texts.append(text); ids.append(r["id"]); meta.append(r)

        if not texts: return
        vecs = self.model.encode(texts, normalize_embeddings=True)
        index_path = f"{self.store_dir}/{ns}.faiss"
        meta_path  = f"{self.store_dir}/{ns}.pkl"

        index = faiss.read_index(index_path) if os.path.exists(index_path) else faiss.IndexFlatIP(self.dim)
        index.add(np.array(vecs))
        faiss.write_index(index, index_path)
        with open(meta_path, "ab") as f:  # append; or maintain a proper KV store
            for m in meta: f.write((json.dumps(m, ensure_ascii=False)+"\n").encode("utf-8"))
