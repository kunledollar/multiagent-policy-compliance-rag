import pickle
from pathlib import Path
from typing import Any, Dict, List

import faiss
import numpy as np

from .config import FAISS_INDEX_PATH, DOCSTORE_PATH, TOP_K

class VectorStore:
    def __init__(self):
        self.index = None
        self.docstore: List[Dict[str, Any]] = []

    def _ensure_index(self, dim: int):
        if self.index is None:
            self.index = faiss.IndexFlatL2(dim)

    def add(self, embeddings: List[List[float]], metadatas: List[Dict[str, Any]]):
        if not embeddings:
            return
        vecs = np.array(embeddings).astype("float32")
        self._ensure_index(vecs.shape[1])
        self.index.add(vecs)
        self.docstore.extend(metadatas)

    def search(self, query_embedding: List[float], k: int = TOP_K) -> List[Dict[str, Any]]:
        if self.index is None or not self.docstore:
            return []
        q = np.array([query_embedding]).astype("float32")
        distances, indices = self.index.search(q, k)
        results: List[Dict[str, Any]] = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx == -1 or idx >= len(self.docstore):
                continue
            meta = dict(self.docstore[idx])
            meta["score"] = float(dist)
            results.append(meta)
        return results

    def save(self):
        if self.index is not None:
            faiss.write_index(self.index, str(FAISS_INDEX_PATH))
        with open(DOCSTORE_PATH, "wb") as f:
            pickle.dump(self.docstore, f)

    def load(self):
        if Path(FAISS_INDEX_PATH).exists():
            self.index = faiss.read_index(str(FAISS_INDEX_PATH))
        if Path(DOCSTORE_PATH).exists():
            with open(DOCSTORE_PATH, "rb") as f:
                self.docstore = pickle.load(f)

vector_store = VectorStore()
