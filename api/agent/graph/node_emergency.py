"""
AxonIQ — Emergency Check Node
Pure regex, runs before any LLM call. No LLM cost for emergencies.
"""
from __future__ import annotations
from api.agent.state import AgentState
from api.agent.tools.emergency import check_emergency


def node_emergency_check(state: AgentState) -> dict:
    result = check_emergency(state["user_message"])
    if result:
        prose = (
            f"⚠️ **{result['label']}**\n\n"
            f"{result['action']}\n\n"
            "Please call emergency services or go to your nearest A&E immediately. "
            "This cannot be assessed through chat — this is a medical emergency."
        )
        return {
            "goal":       "emergency",
            "tier":       "CRITICAL_EMERGENCY",
            "features":   [result["label"]],
            "response":   prose,
            "next_phase": "concluded",  # emergency = concluded immediately
        }
    return {}
