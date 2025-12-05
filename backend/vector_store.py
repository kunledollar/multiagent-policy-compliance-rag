import os
import json
import numpy as np
from pathlib import Path
import faiss

VECTOR_INDEX_PATH = "/app/artifacts/faiss_index.bin"
METADATA_PATH = "/app/artifacts/metadata.json"


class VectorStore:
    def __init__(self):
        self.index = None          # FAISS index
        self.metadatas = []        # List of metadata dicts
        self.dimension = None      # Embedding dimension

    # --------------------------------------------------------
    # LOAD EXISTING INDEX + METADATA
    # --------------------------------------------------------
    def load(self):
        """Load FAISS index + metadata from disk."""
        if os.path.exists(VECTOR_INDEX_PATH):
            self.index = faiss.read_index(VECTOR_INDEX_PATH)
            self.dimension = self.index.d    # Extract vector dimension
        else:
            self.index = None

        if os.path.exists(METADATA_PATH):
            with open(METADATA_PATH, "r") as f:
                self.metadatas = json.load(f)
        else:
            self.metadatas = []

    # --------------------------------------------------------
    # SAVE INDEX + METADATA
    # --------------------------------------------------------
    def save(self):
        """Persist FAISS index + metadata to disk."""
        if self.index is not None:
            faiss.write_index(self.index, VECTOR_INDEX_PATH)

        with open(METADATA_PATH, "w") as f:
            json.dump(self.metadatas, f, indent=2)

    # --------------------------------------------------------
    # CREATE NEW HNSW INDEX
    # --------------------------------------------------------
    def _create_hnsw(self, dim: int):
        index = faiss.index_factory(dim, "HNSW32")
        index.hnsw.efSearch = 64
        index.hnsw.efConstruction = 40
        return index

    # --------------------------------------------------------
    # ADD NEW VECTORS
    # --------------------------------------------------------
    def add(self, vectors, metadata):
        vectors = np.array(vectors).astype("float32")

        # Set dimension if this is the first batch
        if self.index is None:
            self.dimension = vectors.shape[1]
            self.index = self._create_hnsw(self.dimension)

        # Ensure dimensions match
        if vectors.shape[1] != self.dimension:
            raise ValueError("Vector dimension mismatch")

        # Add to FAISS
        self.index.add(vectors)

        # Merge metadata
        self.metadatas.extend(metadata)

    # --------------------------------------------------------
    # SEARCH TOP-K
    # --------------------------------------------------------
    def search(self, query_vector, k=5):
        if self.index is None or len(self.metadatas) == 0:
            return []

        q = np.array(query_vector).astype("float32").reshape(1, -1)

        # FAISS returns (distance, index)
        scores, indices = self.index.search(q, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue  # No result

            item = self.metadatas[idx].copy()
            item["score"] = float(score)
            results.append(item)

        return results

    # --------------------------------------------------------
    # BASIC STATS
    # --------------------------------------------------------
    def stats(self):
        if self.index is None:
            return {"num_vectors": 0}

        return {
            "num_vectors": self.index.ntotal,
            "vector_dim": self.dimension,
        }


# GLOBAL INSTANCE
vector_store = VectorStore()
