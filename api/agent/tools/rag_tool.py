"""
AxonIQ — RAG Tool
Wraps ChromaDB query. Returns context string for the agent.
"""
from __future__ import annotations
from loguru import logger


def fetch_rag_context(query: str) -> str:
    """Query RAG store and return concatenated context string."""
    try:
        from api.rag.store import query as rag_query
        results = rag_query(query)
        return "\n\n".join(r["text"] for r in results)
    except Exception as e:
        logger.warning("[RAG] Error: {}", e)
        return ""
