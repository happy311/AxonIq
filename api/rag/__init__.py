"""Public API for the RAG package."""
from api.rag.store import build_knowledge_base, query

__all__ = ["build_knowledge_base", "query"]
