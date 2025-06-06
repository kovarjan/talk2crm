"""
Load company names & IDs from your MySQL CRM

Create embeddings (e.g., using SentenceTransformers)

Save index + metadata (pickle or DB)
"""

import mysql.connector
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import os
import pickle

# Configuration
DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "Acmark",
    "database": "coripo_localhost"
}

SQL_QUERY = "SELECT id, name FROM accounts WHERE deleted = 0 AND name IS NOT NULL ORDER BY name ASC;"

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384  # for MiniLM

SAVE_DIR = "data/embeddings"
INDEX_FILE = os.path.join(SAVE_DIR, "company_faiss.index")
METADATA_FILE = os.path.join(SAVE_DIR, "company_index.pkl")


def fetch_companies():
    conn = mysql.connector.connect(**DB_CONFIG)
    cursor = conn.cursor()
    cursor.execute(SQL_QUERY)
    rows = cursor.fetchall()
    conn.close()

    company_ids = [str(row[0]) for row in rows]
    company_names = [row[1] for row in rows]
    return company_ids, company_names


def build_index():
    print("🔍 Loading company data from database...")
    company_ids, company_names = fetch_companies()
    print(f"✅ Retrieved {len(company_names)} companies.")

    print("🧠 Loading embedding model:", EMBEDDING_MODEL_NAME)
    model = SentenceTransformer(EMBEDDING_MODEL_NAME, cache_folder="cache/models/sentence_transformers")

    print("🔎 Encoding company names...")
    embeddings = model.encode(company_names, normalize_embeddings=True)

    print("📦 Building FAISS index...")
    index = faiss.IndexFlatIP(EMBEDDING_DIM)
    index.add(np.array(embeddings))

    os.makedirs(SAVE_DIR, exist_ok=True)
    print("💾 Saving index and metadata...")
    faiss.write_index(index, INDEX_FILE)
    with open(METADATA_FILE, "wb") as f:
        pickle.dump({"names": company_names, "ids": company_ids}, f)

    print("✅ Done! Index saved to:")
    print(f"  -> {INDEX_FILE}")
    print(f"  -> {METADATA_FILE}")


if __name__ == "__main__":
    build_index()
