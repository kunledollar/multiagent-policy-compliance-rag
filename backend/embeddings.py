from typing import List
from functools import lru_cache
import openai
from .config import OPENAI_API_KEY, EMBEDDING_MODEL

# ------------------------------------------------------------
# OpenAI Client Initialization
# ------------------------------------------------------------
client = openai.OpenAI(api_key=OPENAI_API_KEY)


# ------------------------------------------------------------
# CACHED SINGLE-EMBEDDING FUNCTION
# ------------------------------------------------------------
@lru_cache(maxsize=2048)
def _cached_single_embedding(text: str) -> List[float]:
    """
    INTERNAL USE ONLY.
    Uses LRU cache to dramatically speed up repeated queries.
    """
    resp = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text,
    )
    return resp.data[0].embedding


def get_embedding(text: str) -> List[float]:
    """
    Public API for single text embedding.
    Safe against blank inputs.
    """
    text = (text or "").strip()
    if not text:
        return []
    return _cached_single_embedding(text)


# ------------------------------------------------------------
# BATCH EMBEDDINGS (Not cached)
# ------------------------------------------------------------
def get_embeddings(texts: List[str]) -> List[List[float]]:
    """
    Generates embeddings for a batch of texts.
    Skips empty strings. Preserves ordering.
    OpenAI handles batching efficiently.
    """
    if not texts:
        return []

    cleaned = [(t or "").strip() for t in texts]

    resp = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=cleaned,
    )

    return [d.embedding for d in resp.data]
