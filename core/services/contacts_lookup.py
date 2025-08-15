from sentence_transformers import SentenceTransformer
import numpy as np, faiss, pickle

class ContactsLookupService:
    def __init__(self):
        with open("data/embeddings/contacts_index.pkl", "rb") as f:
            self.data = pickle.load(f)
        # Use keys from build_contacts_index.py
        first_names = self.data["first_names"]
        last_names = self.data["last_names"]
        account_names = self.data["account_names"]
        self.names = [
            f"{first} {last}, {account}" if account else f"{first} {last}"
            for first, last, account in zip(first_names, last_names, account_names)
        ]
        self.ids = self.data["contact_ids"]
        # No emails/phones in index, fallback to empty
        self.emails = [""] * len(self.names)
        self.phones = [""] * len(self.names)
        self.model = SentenceTransformer("all-MiniLM-L6-v2", cache_folder="cache/models/sentence_transformers")
        self.index = faiss.read_index("data/embeddings/contacts_faiss.index")

    def search(self, query: str, top_k=3):
        emb = self.model.encode([query], normalize_embeddings=True)
        D, I = self.index.search(np.array(emb), top_k)
        results = [
            {
                "name": self.names[i],
                "id": self.ids[i],
                "email": self.emails[i],
                "phone": self.phones[i],
                "score": float(D[0][j])
            }
            for j, i in enumerate(I[0])
        ]
        return results

contacts_lookup_service = ContactsLookupService()

def find_contact_by_name_or_email(query):
    print(f"🤖 [ContactsLookupService] Searching for contact: {query}")
    return contacts_lookup_service.search(query)
