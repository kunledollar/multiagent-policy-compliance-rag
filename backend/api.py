from pathlib import Path
from typing import List

from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from prometheus_fastapi_instrumentator import Instrumentator

# --------------------------------------------
# IMPORT ENTERPRISE LOGGING
# --------------------------------------------
from logging_config import (
    api_logger,
    error_logger,
    new_request_id
)

# --------------------------------------------
# PROJECT IMPORTS
# --------------------------------------------
from .config import DATA_DIR, CHUNK_SIZE, CHUNK_OVERLAP, CHAT_MODEL
from .preprocess import clean_text, chunk_text
from .embeddings import get_embeddings
from .vector_store import vector_store
from .rag_orchestrator import answer_query


# --------------------------------------------
# FASTAPI APP
# --------------------------------------------
app = FastAPI(title="Enterprise Policy & Compliance Assistant — Multi-Agent RAG System")


# ---------------------------------------------------
# ENTERPRISE REQUEST LOGGING MIDDLEWARE
# ---------------------------------------------------
@app.middleware("http")
async def log_requests(request: Request, call_next):
    request_id = new_request_id()

    # Log incoming request
    api_logger.info(
        f"Incoming {request.method} request to {request.url.path}",
        extra={"request_id": request_id}
    )

    try:
        response = await call_next(request)
    except Exception as e:
        # Log errors with stack trace
        error_logger.error(
            f"Unhandled error: {str(e)}",
            extra={"request_id": request_id}
        )
        raise

    # Log response
    api_logger.info(
        f"Completed request with status {response.status_code}",
        extra={"request_id": request_id}
    )

    return response


# ---------------------------------------------------
# REQUEST / RESPONSE MODELS
# ---------------------------------------------------
class IngestRequest(BaseModel):
    directory: str | None = None


class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    answer: str
    contexts: List[dict]
    reasoning: str
    fact_check: str
    sources: List[str]


# ---------------------------------------------------
# STARTUP: LOAD VECTOR INDEX & ENABLE METRICS
# ---------------------------------------------------
@app.on_event("startup")
def startup_event():
    api_logger.info("Starting API... loading vector index (if available).")

    vector_store.load()
    Instrumentator().instrument(app).expose(app)

    api_logger.info("API startup complete.")


# ---------------------------------------------------
# HEALTHCHECK ENDPOINT
# ---------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok"}


# ---------------------------------------------------
# INGEST ENDPOINT
# ---------------------------------------------------
@app.post("/ingest")
def ingest(req: IngestRequest):
    directory = Path(req.directory) if req.directory else DATA_DIR
    if not directory.exists():
        raise HTTPException(status_code=400, detail="Ingest directory does not exist")

    texts: List[str] = []
    metadatas: List[dict] = []

    supported_ext = {".txt", ".md", ".pdf", ".docx"}
    api_logger.info(f"Starting ingestion from {directory}")

    for file_path in directory.rglob("*"):
        if file_path.suffix.lower() not in supported_ext:
            continue

        try:
            # Extract text from supported file types
            if file_path.suffix.lower() in {".txt", ".md"}:
                raw = file_path.read_text(encoding="utf-8", errors="ignore")

            elif file_path.suffix.lower() == ".pdf":
                from pypdf import PdfReader
                reader = PdfReader(str(file_path))
                raw = "\n".join(page.extract_text() or "" for page in reader.pages)

            elif file_path.suffix.lower() == ".docx":
                import docx
                doc = docx.Document(str(file_path))
                raw = "\n".join(p.text for p in doc.paragraphs)

            else:
                continue

        except Exception as e:
            error_logger.error(
                f"Failed to read file {file_path}: {e}",
                extra={"request_id": new_request_id()}
            )
            continue

        clean = clean_text(raw)
        chunks = chunk_text(clean, CHUNK_SIZE, CHUNK_OVERLAP)
        policy_id = file_path.stem.split("_")[0]

        for i, chunk in enumerate(chunks):
            texts.append(chunk)
            metadatas.append({
                "source": str(file_path),
                "policy_id": policy_id,
                "chunk_id": i,
                "text": chunk,
            })

    if not texts:
        raise HTTPException(status_code=400, detail="No supported files found for ingestion")

    api_logger.info(f"Generating embeddings for {len(texts)} chunks")
    embeddings = get_embeddings(texts)
    vector_store.add(embeddings, metadatas)
    vector_store.save()

    api_logger.info(f"Ingestion complete: {len(texts)} chunks added.")
    return {"status": "ok", "num_chunks": len(texts)}


# ---------------------------------------------------
# RAG QUERY ENDPOINT
# ---------------------------------------------------
@app.post("/query", response_model=QueryResponse)
def query_rag(req: QueryRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    api_logger.info(f"Received query: {req.query}")

    result = answer_query(req.query)

    return QueryResponse(
        answer=result["answer"],
        contexts=result["contexts"],
        reasoning=result["reasoning"],
        fact_check=result["fact_check"],
        sources=result["sources"],
    )
