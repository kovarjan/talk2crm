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

SQL_QUERY = "SELECT c.id, c.first_name, c.last_name, a.name AS account_name, a.id AS account_id FROM contacts AS c" \
" LEFT JOIN accounts_contacts AS ac ON c.id = ac.contact_id" \
" LEFT JOIN accounts AS a ON ac.account_id = a.id" \
" WHERE c.deleted = 0 ORDER BY c.first_name ASC;"

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384  # for MiniLM

SAVE_DIR = "data/embeddings"
INDEX_FILE = os.path.join(SAVE_DIR, "contacts_faiss.index")
METADATA_FILE = os.path.join(SAVE_DIR, "contacts_index.pkl")

def fetch_contacts():
    conn = mysql.connector.connect(**DB_CONFIG)
    cursor = conn.cursor()
    cursor.execute(SQL_QUERY)
    rows = cursor.fetchall()
    conn.close()

    contact_ids = [str(row[0]) for row in rows]
    first_names = [row[1] for row in rows]
    last_names = [row[2] for row in rows]
    account_names = [row[3] for row in rows]
    account_ids = [str(row[4]) if row[4] is not None else "" for row in rows]
    # Combine first name, last name, and account name for context
    return contact_ids, first_names, last_names, account_names, account_ids


def build_index():
    print("🔍 Loading contact data from database...")
    contact_ids, first_names, last_names, account_names, account_ids = fetch_contacts()
    print(f"✅ Retrieved {len(first_names)} contacts.")

    print("🧠 Loading embedding model:", EMBEDDING_MODEL_NAME)
    model = SentenceTransformer(EMBEDDING_MODEL_NAME, cache_folder="cache/models/sentence_transformers")

    print("🔎 Encoding contact names...")
    # Combine first name, last name, and account name for richer context
    texts = [
        f"{first} {last}, {account}" if account else f"{first} {last}"
        for first, last, account in zip(first_names, last_names, account_names)
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
        pickle.dump({
            "contact_ids": contact_ids,
            "first_names": first_names,
            "last_names": last_names,
            "account_names": account_names,
            "account_ids": account_ids
        }, f)

    print("✅ Done! Index saved to:")
    print(f"  -> {INDEX_FILE}")
    print(f"  -> {METADATA_FILE}")


if __name__ == "__main__":
    build_index()
