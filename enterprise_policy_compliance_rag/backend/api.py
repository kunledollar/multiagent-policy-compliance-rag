from pathlib import Path
from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from prometheus_fastapi_instrumentator import Instrumentator

from .config import DATA_DIR, CHUNK_SIZE, CHUNK_OVERLAP
from .logging_config import setup_logging
from .preprocess import clean_text, chunk_text
from .embeddings import get_embeddings
from .vector_store import vector_store
from .rag_orchestrator import answer_query

import logging
logger = setup_logging()

app = FastAPI(title="Enterprise Policy & Compliance Assistant — Multi-Agent RAG System")

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

@app.on_event("startup")
def startup_event():
    logger.info("Loading vector index (if present)...")
    vector_store.load()
    Instrumentator().instrument(app).expose(app)
    logger.info("API startup complete.")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/ingest")
def ingest(req: IngestRequest):
    directory = Path(req.directory) if req.directory else DATA_DIR
    if not directory.exists():
        raise HTTPException(status_code=400, detail="Ingest directory does not exist")

    texts: List[str] = []
    metadatas: List[dict] = []

    supported_ext = {".txt", ".md", ".pdf", ".docx"}
    logger.info("Starting ingestion from %s", directory)

    for file_path in directory.rglob("*"):
        if file_path.suffix.lower() not in supported_ext:
            continue
        try:
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
            logger.exception("Failed to read file %s: %s", file_path, e)
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

    logger.info("Generating embeddings for %d chunks", len(texts))
    embeddings = get_embeddings(texts)
    vector_store.add(embeddings, metadatas)
    vector_store.save()
    logger.info("Ingestion complete: %d chunks", len(texts))

    return {"status": "ok", "num_chunks": len(texts)}

@app.post("/query", response_model=QueryResponse)
def query_rag(req: QueryRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    logger.info("Received query: %s", req.query)
    result = answer_query(req.query)
    return QueryResponse(
        answer=result["answer"],
        contexts=result["contexts"],
        reasoning=result["reasoning"],
        fact_check=result["fact_check"],
        sources=result["sources"],
    )
