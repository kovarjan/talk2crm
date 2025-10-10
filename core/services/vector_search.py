from __future__ import annotations

import os, pickle, json, gzip
from typing import List, Dict, Any, Optional

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


# ---------- shared model (singleton) ----------

class _Model:
    _inst: Optional[SentenceTransformer] = None

    @classmethod
    def get(cls) -> SentenceTransformer:
        if cls._inst is None:
            # keep the same model you used for ingestion
            cls._inst = SentenceTransformer("all-MiniLM-L6-v2", cache_folder="cache/models/sentence_transformers")
        return cls._inst


# ---------- metadata loader (pickle OR jsonl) ----------

def _load_metadata(meta_path: str) -> List[Dict[str, Any]]:
    """
    Returns a list of records (aligned 1:1 with vectors in the FAISS index).
    Tries pickle first; if that fails, treats the file as JSONL (optionally gzipped).
    """
    if not os.path.exists(meta_path):
        raise FileNotFoundError(meta_path)

    # Try pickle
    try:
        with open(meta_path, "rb") as f:
            obj = pickle.load(f)
        # common shapes: list[dict] OR {"records":[...]}
        if isinstance(obj, dict) and "records" in obj:
            return list(obj["records"])
        if isinstance(obj, list):
            return obj
    except Exception:
        pass

    # Try JSONL (plain or gz)
    def _iter_lines(path: str):
        if path.endswith(".gz"):
            with gzip.open(path, "rt", encoding="utf-8") as f:
                for line in f: yield line
        else:
            with open(path, "rt", encoding="utf-8") as f:
                for line in f: yield line

    records: List[Dict[str, Any]] = []
    try:
        for line in _iter_lines(meta_path):
            line = line.strip()
            if not line: continue
            records.append(json.loads(line))
        if records:
            return records
    except Exception:
        pass

    raise ValueError(f"Unsupported metadata format: {meta_path}")


# ---------- generic vector searcher ----------

class VectorSearcher:
    """
    Opens a FAISS index + metadata for a given tenant+module namespace.
    Namespace files:
      var/vector/{tenant}:{module}.faiss
      var/vector/{tenant}:{module}.pkl
    """
    def __init__(self, tenant: str, module: str, vector_dir: str = "var/vector"):
        self.ns = f"{tenant}:{module}"
        self.faiss_path = os.path.join(vector_dir, f"{self.ns}.faiss")
        self.meta_path  = os.path.join(vector_dir, f"{self.ns}.pkl")  # may be pickle OR jsonl
        if not os.path.exists(self.faiss_path):
            raise FileNotFoundError(self.faiss_path)
        if not os.path.exists(self.meta_path):
            # allow .jsonl or .jsonl.gz fallback if you opted for that during ingest
            alt_jsonl = self.meta_path[:-4] + ".jsonl"
            alt_gz    = alt_jsonl + ".gz"
            if os.path.exists(alt_gz): self.meta_path = alt_gz
            elif os.path.exists(alt_jsonl): self.meta_path = alt_jsonl
            else: raise FileNotFoundError(self.meta_path)

        self.index = faiss.read_index(self.faiss_path)
        self.meta  = _load_metadata(self.meta_path)
        self.model = _Model.get()

        # quick sanity: counts should roughly match (FAISS keeps ntotal)
        if hasattr(self.index, "ntotal") and self.index.ntotal and len(self.meta) and self.index.ntotal != len(self.meta):
            # not fatal, but warn: ingestion may have appended vecs without metadata flush
            pass

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        emb = self.model.encode([query], normalize_embeddings=True)
        D, I = self.index.search(np.array(emb), top_k)
        out: List[Dict[str, Any]] = []
        for rank, idx in enumerate(I[0]):
            if idx < 0 or idx >= len(self.meta):  # FAISS can return -1 if empty
                continue
            score = float(D[0][rank])
            out.append({"_index": int(idx), "_score": score, **self.meta[idx]})
        return out
