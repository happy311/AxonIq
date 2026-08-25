"""
Node: RAG Retrieval
Fetches clinical context from ChromaDB based on current symptoms + user message.
"""
from __future__ import annotations
from api.agent.state import AgentState
from api.agent.tools.rag_tool import fetch_rag_context


def node_rag_retrieval(state: AgentState) -> dict:
    query = state["user_message"]
    # Enrich query with known features for better RAG hits
    features = state.get("features", [])
    if features:
        query = query + " " + " ".join(features[:3])

    context = fetch_rag_context(query)
    return {"rag_context": context}
