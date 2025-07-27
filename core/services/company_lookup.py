from sentence_transformers import SentenceTransformer
import numpy as np, faiss, pickle

class CompanyLookupService:
    def __init__(self):
        with open("data/embeddings/company_index.pkl", "rb") as f:
            self.data = pickle.load(f)
        self.names = self.data["names"]
        self.ids = self.data["ids"]
        self.cities = self.data.get("cities", [""] * len(self.names))  # fallback if not present
        self.model = SentenceTransformer("all-MiniLM-L6-v2", cache_folder="cache/models/sentence_transformers")
        self.index = faiss.read_index("data/embeddings/company_faiss.index")

    def search(self, query: str, top_k=3):
        emb = self.model.encode([query], normalize_embeddings=True)
        D, I = self.index.search(np.array(emb), top_k)
        results = [
            {
                "name": self.names[i],
                "id": self.ids[i],
                "city": self.cities[i],
                "score": float(D[0][j])
            }
            for j, i in enumerate(I[0])
        ]
        return results

company_lookup_service = CompanyLookupService()

def find_company_by_name(name_or_city):
    return company_lookup_service.search(name_or_city)
