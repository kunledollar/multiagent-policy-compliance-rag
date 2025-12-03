from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "raw"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
LOG_DIR = PROJECT_ROOT / "logs"

for d in (DATA_DIR, ARTIFACTS_DIR, LOG_DIR):
    d.mkdir(parents=True, exist_ok=True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4.1-mini")

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
TOP_K = int(os.getenv("TOP_K", "6"))

FAISS_INDEX_PATH = ARTIFACTS_DIR / "faiss_index.bin"
DOCSTORE_PATH = ARTIFACTS_DIR / "docstore.pkl"

RAG_SYSTEM_PROMPT = (
    "You are an enterprise policy and compliance assistant. "
    "Answer strictly based on the provided context. "
    "If the answer is not present, say that the policy does not specify or that you do not know. "
    "Always mention which policy documents you used."
)
