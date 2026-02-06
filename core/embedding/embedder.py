from sentence_transformers import SentenceTransformer
import faiss, numpy as np, pickle, os, json, re, logging

class Embedder:
    def __init__(self, model_name="all-MiniLM-L6-v2", dim=384, store_dir="var/vector", log_embed: bool = False, log_dir: str = "logs"):
        self.model = SentenceTransformer(model_name)
        self.dim, self.store_dir = dim, store_dir
        os.makedirs(store_dir, exist_ok=True)
        self.logger = None
        if log_embed:
            os.makedirs(log_dir, exist_ok=True)
            log_path = os.path.join(log_dir, "ingest_embed.log")
            logger = logging.getLogger("embedder")
            if not logger.handlers:
                handler = logging.FileHandler(log_path)
                formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
                handler.setFormatter(formatter)
                logger.addHandler(handler)
            logger.setLevel(logging.INFO)
            logger.propagate = False
            self.logger = logger
        self._legal_suffixes = (
            "a.s",
            "a.s.",
            "s.r.o",
            "s.r.o.",
            "spol. s r.o.",
            "spol s r.o.",
            "inc",
            "inc.",
            "ltd",
            "ltd.",
            "llc",
            "llc.",
            "gmbh",
            "s.a.",
            "s.a",
            "s.p.a.",
            "s.p.a",
            "plc",
            "plc.",
            "corp",
            "corp.",
            "co",
            "co.",
        )

    def simplify_company_name(self, name: str) -> str:
        if not name:
            return ""
        cleaned = name.lower()
        cleaned = re.sub(r"[\"'`]", "", cleaned)
        for suffix in self._legal_suffixes:
            cleaned = re.sub(rf"\b{re.escape(suffix)}\b", " ", cleaned)
        cleaned = re.sub(r"[^\w\s]", " ", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned.replace(" ", "")

    def normalize_text(self, text: str) -> str:
        if not text:
            return ""
        return re.sub(r"\s+", " ", text).strip()

    def build_text_accounts(self, rec, fields):
        f = rec["fields"]
        # Start with the main fields
        simple_name = self.simplify_company_name(f.get("name", ""))
        parts = [f.get("name",""), simple_name, f.get("industry",""), f.get("account_type",""),
             f.get("phone_office",""), f.get("email1",""),
             f.get("billing_address_city",""), f.get("billing_address_country",""),
             f.get("description","")]
        # Add all other fields not already included
        for key, value in f.items():
            if key not in {"name", "industry", "account_type", "phone_office", "email1", "billing_address_city", "billing_address_country", "description"} and value:
                parts.append(key + ": " + str(value))
        return ", ".join([p for p in parts if p])

    def build_text_contacts(self, rec, fields):
        f = rec["fields"]
        name = " ".join([f.get("first_name",""), f.get("last_name","")]).strip()
        # Start with the main fields
        parts = [name, f.get("title",""), f.get("department",""), f.get("email1",""), f.get("phone_mobile",""), f.get("description","")]
        # Add user field
        if f.get("assigned_user_id"):
            parts.append("user: " + str(f.get("assigned_user_id")))
        # Add all other fields not already included
        main_keys = {"first_name", "last_name", "full_name", "title", "department", "email1", "phone_mobile", "description"}
        for key, value in f.items():
            if key not in main_keys and value:
                parts.append(key + ": " + str(value))
        return ", ".join([p for p in parts if p])
    
    def build_text_meetings(self, rec, fields):
        f = rec["fields"]
        # Start with the main fields
        parts = [f.get("name",""), f.get("date_start",""), f.get("date_end",""),
                 f.get("status",""), f.get("location",""), f.get("description",""),
                 f.get("parent_type",""), f.get("parent_id","")]
        # Add all other fields not already included
        main_keys = {"name", "date_start", "date_end", "status", "location", "description", "parent_type", "parent_id"}
        for key, value in f.items():
            if key not in main_keys and value:
                parts.append(key + ": " + str(value))
        return ", ".join([p for p in parts if p])

    def _generic_text(self, rec, fields):
        f = rec.get("fields", {})
        parts = [str(f.get(k, "")) for k in fields if f.get(k, "")]
        return ", ".join(parts)

    def upsert_batch(self, tenant: str, module: str, records: list, fields: list):
        ns = f"{tenant}:{module}"
        texts, ids, meta = [], [], []
        for r in records:
            if r.get("deleted"): continue
            if module == "accounts": text = self.build_text_accounts(r, fields)
            elif module == "contacts": text = self.build_text_contacts(r, fields)
            elif module == "meetings": text = self.build_text_meetings(r, fields)
            else: text = self._generic_text(r, fields)
            text = self.normalize_text(text)
            print("[embed] Embedding text:", text)
            if self.logger:
                self.logger.info("tenant=%s module=%s id=%s text=%s", tenant, module, r.get("id"), text)
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
