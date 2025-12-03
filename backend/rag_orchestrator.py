from typing import Dict, Any

from .config import TOP_K
from .agents import (
    retrieval_agent,
    reranker_agent,
    summarizer_agent,
    compliance_reasoner_agent,
    fact_checker_agent,
    answer_writer_agent,
)

def answer_query(query: str) -> Dict[str, Any]:
    retrieved = retrieval_agent(query, top_k=TOP_K)
    if not retrieved:
        return {
            "answer": "I could not find any relevant policy excerpts for this question.",
            "contexts": [],
            "reasoning": "",
            "fact_check": "No documents retrieved",
            "sources": [],
        }

    reranked = reranker_agent(query, retrieved)
    summary = summarizer_agent(reranked)
    reasoning = compliance_reasoner_agent(query, summary)
    fact_verdict, sources = fact_checker_agent(query, reasoning, reranked)
    answer = answer_writer_agent(query, reranked, reasoning, fact_verdict)

    return {
        "answer": answer,
        "contexts": reranked,
        "reasoning": reasoning,
        "fact_check": fact_verdict,
        "sources": sources,
    }
