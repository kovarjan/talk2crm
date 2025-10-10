"""
NOTE: Deprecated - replaced by from core.ingestion.ingestor import Ingestor and from core.embedding.embedder import Embedder

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

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384  # for MiniLM

SAVE_DIR = "data/embeddings"
INDEX_FILE = os.path.join(SAVE_DIR, "company_faiss.index")
METADATA_FILE = os.path.join(SAVE_DIR, "company_index.pkl")

SQL_QUERY = """
SELECT
    id,
    name,
    industry,
    account_type,
    annual_revenue,
    employees,
    website,
    billing_address_city,
    billing_address_country,
    description
FROM accounts
WHERE deleted = 0 AND name IS NOT NULL
ORDER BY name ASC;
"""

def fetch_companies():
    conn = mysql.connector.connect(**DB_CONFIG)
    cursor = conn.cursor()
    cursor.execute(SQL_QUERY)
    rows = cursor.fetchall()
    conn.close()

    # Unpack columns
    (
        company_ids,
        company_names,
        industries,
        account_types,
        annual_revenues,
        employees,
        websites,
        billing_address_citys,
        billing_address_countries,
        descriptions
    ) = zip(*rows) if rows else ([], [], [], [], [], [], [], [], [], [])

    return {
        "ids": [str(x) for x in company_ids],
        "names": list(company_names),
        "industries": list(industries),
        "account_types": list(account_types),
        "annual_revenues": list(annual_revenues),
        "employees": list(employees),
        "websites": list(websites),
        "cities": list(billing_address_citys),
        "countries": list(billing_address_countries),
        "descriptions": list(descriptions)
    }

def build_index():
    print("🔍 Loading company data from database...")
    data = fetch_companies()
    print(f"✅ Retrieved {len(data['names'])} companies.")

    print("🧠 Loading embedding model:", EMBEDDING_MODEL_NAME)
    model = SentenceTransformer(EMBEDDING_MODEL_NAME, cache_folder="cache/models/sentence_transformers")

    print("🔎 Encoding company profiles...")
    # Combine selected fields for richer context
    texts = [
        ", ".join([
            str(data["names"][i] or ""),
            str(data["industries"][i] or ""),
            str(data["account_types"][i] or ""),
            str(data["annual_revenues"][i] or ""),
            str(data["employees"][i] or ""),
            str(data["websites"][i] or ""),
            str(data["cities"][i] or ""),
            str(data["countries"][i] or ""),
            str(data["descriptions"][i] or "")
        ]).strip(", ")
        for i in range(len(data["names"]))
    ]

    print(f"📏 Encoding {len(texts)} texts with {EMBEDDING_DIM}-dimensional embeddings...")
    embeddings = model.encode(texts, normalize_embeddings=True)

    print("📦 Building FAISS index...")
    index = faiss.IndexFlatIP(EMBEDDING_DIM)
    index.add(np.array(embeddings))

    os.makedirs(SAVE_DIR, exist_ok=True)
    print("💾 Saving index and metadata...")
    faiss.write_index(index, INDEX_FILE)
    with open(METADATA_FILE, "wb") as f:
        pickle.dump(data, f)

    print("✅ Done! Index saved to:")
    print(f"  -> {INDEX_FILE}")
    print(f"  -> {METADATA_FILE}")

if __name__ == "__main__":
    build_index()
