"""
This file will:

Load & cache your embedding index (using FAISS)

Provide find_company_by_name(query: str) to use in LangChain

Optionally expose a Tool for LangChain Agents
"""

from sentence_transformers import SentenceTransformer
import numpy as np, faiss, pickle

class CompanyLookupService:
    def __init__(self):
        with open("data/embeddings/company_index.pkl", "rb") as f:
            self.data = pickle.load(f)
        self.names = self.data["names"]
        self.ids = self.data["ids"]
        self.model = SentenceTransformer("all-MiniLM-L6-v2", cache_folder="cache/models/sentence_transformers")
        self.index = faiss.read_index("data/embeddings/company_faiss.index")

    def search(self, query: str, top_k=3):
        emb = self.model.encode([query], normalize_embeddings=True)
        D, I = self.index.search(np.array(emb), top_k)
        results = [{"name": self.names[i], "id": self.ids[i], "score": float(D[0][j])} for j, i in enumerate(I[0])]
        return results

company_lookup_service = CompanyLookupService()

def find_company_by_name(name):
    return company_lookup_service.search(name)

