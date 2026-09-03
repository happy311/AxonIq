"""
AxonIQ — Graph Edges

SYSTEM DESIGN: Routing is purely based on state fields set by nodes.
No string matching on LLM output anywhere in this file.
"""
from __future__ import annotations
from api.agent.state import AgentState


def route_after_emergency(state: AgentState) -> str:
    """If emergency node wrote a response → skip everything → end."""
    if state.get("goal") == "emergency":
        return "llm_response"   # canned safety message from node_emergency_check is passed through verbatim by node_llm_response
    return "goal_setter"


def route_after_goal(state: AgentState) -> str:
    """mri_received / mri_recheck go to mri_analysis node first; everything
    else → rag → llm. mri_recheck (v17) is a patient-triggered "check the mri
    again" — node_mri._handle_recheck does a real single-shot server fetch."""
    if state.get("goal") in ("mri_received", "mri_recheck"):
        return "mri_analysis"
    return "rag_retrieval"


def route_after_mri(state: AgentState) -> str:
    return "rag_retrieval"


def route_after_llm(state: AgentState) -> str:
    return "end"
