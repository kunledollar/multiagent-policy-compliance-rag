from typing import List
import openai
from .config import OPENAI_API_KEY, EMBEDDING_MODEL

client = openai.OpenAI(api_key=OPENAI_API_KEY)

def get_embedding(text: str) -> List[float]:
    if not text.strip():
        return []
    resp = client.embeddings.create(
        input=text,
        model=EMBEDDING_MODEL,
    )
    return resp.data[0].embedding

def get_embeddings(texts: List[str]) -> List[List[float]]:
    if not texts:
        return []
    resp = client.embeddings.create(
        input=texts,
        model=EMBEDDING_MODEL,
    )
    return [d.embedding for d in resp.data]
