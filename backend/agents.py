from typing import Dict, List, Tuple
import openai

from .config import CHAT_MODEL, RAG_SYSTEM_PROMPT
from .embeddings import get_embedding
from .vector_store import vector_store

# 🔥 Logging imports
from logging_config import (
    query_agent_logger,
    retriever_agent_logger,
    synth_agent_logger,
    pipeline_logger
)

client = openai.OpenAI()

# ----------------------------------------------------
# RETRIEVAL AGENT
# ----------------------------------------------------
def retrieval_agent(query: str, top_k: int) -> List[Dict]:
    retriever_agent_logger.info(
        f"Retrieval agent received query='{query}', top_k={top_k}",
        extra={"agent": "retrieval"}
    )

    q_emb = get_embedding(query)
    return vector_store.search(q_emb, k=top_k)


# ----------------------------------------------------
# RERANKER AGENT
# ----------------------------------------------------
def reranker_agent(query: str, candidates: List[Dict]) -> List[Dict]:
    pipeline_logger.info(
        f"Reranker agent sorting {len(candidates)} candidates for query='{query}'",
        extra={"pipeline_step": "rerank"}
    )

    return sorted(candidates, key=lambda x: x.get("score", 0.0))


# ----------------------------------------------------
# SUMMARIZER AGENT
# ----------------------------------------------------
def summarizer_agent(chunks: List[Dict]) -> str:
    synth_agent_logger.info(
        f"Summarizer agent condensing {len(chunks)} chunks",
        extra={"agent": "summarizer"}
    )

    pieces = []
    for c in chunks:
        src = c.get("policy_id") or c.get("source", "unknown")
        pieces.append(f"[{src}] {c.get('text', '')}")

    joined = "\n".join(pieces)[:6000]

    msg = (
        "Summarise the following policy excerpts into key bullet points focused on "
        "rules, thresholds, timelines, and obligations. Preserve any numbers or limits.\n\n" 
        + joined
    )

    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": msg}],
        temperature=0.2,
    )
    return resp.choices[0].message.content


# ----------------------------------------------------
# COMPLIANCE REASONER AGENT
# ----------------------------------------------------
def compliance_reasoner_agent(query: str, summary: str) -> str:
    pipeline_logger.info(
        f"Compliance reasoning agent analyzing query='{query}'",
        extra={"pipeline_step": "compliance_reasoning"}
    )

    prompt = (
        f"Question: {query}\n\n"
        f"Policy summary:\n{summary}\n\n"
        "Answer the question as a senior compliance officer. "
        "Identify which rules apply, which do not, and where there is ambiguity."
    )

    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return resp.choices[0].message.content


# ----------------------------------------------------
# FACT CHECKER AGENT
# ----------------------------------------------------
def fact_checker_agent(query: str, answer: str, chunks: List[Dict]) -> Tuple[str, List[str]]:
    pipeline_logger.info(
        f"Fact-checker agent validating answer for query='{query}' using {len(chunks)} chunks",
        extra={"pipeline_step": "fact_check"}
    )

    context = "\n".join(c.get("text", "") for c in chunks)[:6000]

    prompt = (
        "You are a strict compliance fact checker.\n"
        f"User question: {query}\n"
        f"Proposed answer: {answer}\n"
        f"Context from policy documents:\n{context}\n\n"
        "Identify any parts of the answer that are not directly supported by the context. "
        "If everything is supported, say 'Answer fully supported'. "
        "Otherwise, list unsupported or speculative claims."
    )

    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )

    verdict = resp.choices[0].message.content
    return verdict, [c.get("source", "") for c in chunks]


# ----------------------------------------------------
# FINAL ANSWER WRITER (SYNTHESIZER)
# ----------------------------------------------------
def answer_writer_agent(query: str, chunks: List[Dict], reasoning: str, fact_check_verdict: str) -> str:
    synth_agent_logger.info(
        f"Answer writer agent generating final answer for query='{query}'",
        extra={"agent": "answer_writer"}
    )

    context_strs = []
    for c in chunks:
        src = c.get("policy_id") or c.get("source", "unknown")
        context_strs.append(f"[Source: {src}] Extract: {c.get('text','')}")
    context_block = "\n\n".join(context_strs)[:6000]

    messages = [
        {"role": "system", "content": RAG_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Question: {query}\n\n"
                f"Relevant policy context:\n{context_block}\n\n"
                f"Internal compliance reasoning:\n{reasoning}\n\n"
                f"Fact-check verdict:\n{fact_check_verdict}\n\n"
                "Write a clear, concise answer for a business stakeholder. "
                "Cite policy IDs or titles where relevant."
            )
        }
    ]

    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        temperature=0.2,
    )

    return resp.choices[0].message.content
