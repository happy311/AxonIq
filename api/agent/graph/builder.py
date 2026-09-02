"""
AxonIQ — LangGraph Builder

Graph topology (one pass per user turn):

  [START]
     │
     ▼
  emergency_check ──(emergency)──► llm_response ──► [END]
     │ (no emergency)
     ▼
  goal_setter
     │
     ├──(mri_received / mri_recheck)──► mri_analysis ──► rag_retrieval ──► llm_response ──► [END]
     │
     └──(all other goals)──► rag_retrieval ──► llm_response ──► [END]
"""
from __future__ import annotations
from langgraph.graph import StateGraph, END

from api.agent.state import AgentState
from api.agent.graph.node_emergency  import node_emergency_check
from api.agent.graph.node_goal_setter import node_goal_setter
from api.agent.graph.node_rag        import node_rag_retrieval
from api.agent.graph.node_mri        import node_mri_analysis
from api.agent.graph.node_llm        import node_llm_response
from api.agent.graph.edges import (
    route_after_emergency,
    route_after_goal,
    route_after_mri,
    route_after_llm,
)

_graph = None


def build_graph():
    g = StateGraph(AgentState)

    g.add_node("emergency_check", node_emergency_check)
    g.add_node("goal_setter",     node_goal_setter)
    g.add_node("rag_retrieval",   node_rag_retrieval)
    g.add_node("mri_analysis",    node_mri_analysis)
    g.add_node("llm_response",    node_llm_response)

    g.set_entry_point("emergency_check")

    g.add_conditional_edges(
        "emergency_check",
        route_after_emergency,
        {"llm_response": "llm_response", "goal_setter": "goal_setter"},
    )
    g.add_conditional_edges(
        "goal_setter",
        route_after_goal,
        {"mri_analysis": "mri_analysis", "rag_retrieval": "rag_retrieval"},
    )
    g.add_conditional_edges(
        "mri_analysis",
        route_after_mri,
        {"rag_retrieval": "rag_retrieval"},
    )
    g.add_edge("rag_retrieval", "llm_response")
    g.add_conditional_edges(
        "llm_response",
        route_after_llm,
        {"end": END},
    )

    return g.compile()


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph
