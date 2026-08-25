"""
AxonIQ — Agent Entrypoint (v11)

SYSTEM DESIGN:
- Phase is the single source of truth (stored in DB, passed in from route)
- No string matching on history to infer state
- McDonald criteria (dis_regions, dit_episodes, symptom_timeline) are persisted
  per turn alongside tier and features

chat() returns a dict — cleaner than a growing tuple.
generate_summary() calls the LLM with a summary prompt for the /export endpoint.
"""
from __future__ import annotations
from typing import List, Dict, Optional
from loguru import logger

from api.agent.graph.builder import get_graph
from api.agent.state import AgentState


def chat(
    session_id:       str,
    user_message:     str,
    history:          List[Dict],
    phase:            str,
    tier:             str,
    features:         List[str],
    nifti_paths:      Optional[dict] = None,   # {"flair": path} — MRI backend is FLAIR-only
    dis_regions:      List[str]     = None,
    dit_episodes:     int           = 0,
    symptom_timeline: List[dict]    = None,
) -> dict:
    """
    Run one turn of the agentic graph.

    Returns dict with keys:
      prose, tier, features, next_phase, dis_regions, dit_episodes, symptom_timeline
    """
    dis_regions      = dis_regions      or []
    symptom_timeline = symptom_timeline or []

    human_turns = sum(1 for m in history if m["role"] == "human")

    initial_state: AgentState = {
        "session_id":       session_id,
        "user_message":     user_message,
        "history":          history,
        "human_turns":      human_turns,

        # Phase from DB — goal setter reads this
        "phase":            phase,

        # Classification (from DB — accumulated across turns)
        "tier":             tier,
        "features":         features,

        # McDonald criteria (from DB — accumulated across turns)
        "dis_regions":      dis_regions,
        "dit_episodes":     dit_episodes,
        "symptom_timeline": symptom_timeline,

        # MRI
        "mri_report":        None,
        "nifti_paths":       nifti_paths,
        "mri_service_failed": False,

        # Outputs (set by nodes during graph execution)
        "goal":             "gathering",
        "rag_context":      "",
        "response":         "",
        "next_phase":       phase,  # default: stay in current phase
    }

    try:
        graph  = get_graph()
        result = graph.invoke(initial_state)
    except Exception as e:
        logger.error("[chat] Graph error: {}", e)
        return {
            "prose":            "I'm having trouble processing your message. Please try again.",
            "tier":             tier,
            "features":         features,
            "next_phase":       phase,
            "dis_regions":      dis_regions,
            "dit_episodes":     dit_episodes,
            "symptom_timeline": symptom_timeline,
        }

    prose            = result.get("response", "") or "Could you tell me more about your symptoms?"
    new_tier         = result.get("tier",             tier)
    new_features     = result.get("features",         features)
    next_phase       = result.get("next_phase",       phase)
    new_dis          = result.get("dis_regions",      dis_regions)
    new_dit          = result.get("dit_episodes",     dit_episodes)
    new_timeline     = result.get("symptom_timeline", symptom_timeline)

    logger.info(
        "[chat] session={} turn={} phase={}→{} tier={}→{} dis={}",
        session_id, human_turns, phase, next_phase, tier, new_tier, new_dis,
    )

    return {
        "prose":            prose,
        "tier":             new_tier,
        "features":         new_features,
        "next_phase":       next_phase,
        "dis_regions":      new_dis,
        "dit_episodes":     new_dit,
        "symptom_timeline": new_timeline,
    }


def generate_summary(
    tier:             str,
    features:         List[str],
    symptom_timeline: List[dict],
    dis_regions:      List[str],
    dit_episodes:     int,
) -> dict:
    """
    Generate a structured clinical summary for the /export endpoint.
    Calls the LLM with the SUMMARY_GOAL prompt.
    Returns a structured dict, or a minimal fallback if the LLM fails.
    """
    from api.agent.llm import llm
    from api.agent.prompts import summary as summary_prompt
    from langchain_core.messages import SystemMessage, HumanMessage
    import json, re

    prompt = summary_prompt(tier, features, symptom_timeline, dis_regions, dit_episodes)

    try:
        resp = llm.invoke([
            SystemMessage(content="You are a clinical AI that produces structured JSON summaries. Output ONLY valid JSON."),
            HumanMessage(content=prompt),
        ])
        raw = resp.content if hasattr(resp, "content") else str(resp)
        # Strip any markdown fences
        raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
        return json.loads(raw)
    except Exception as e:
        logger.error("[Summary] Failed to generate summary: {}", e)
        return {
            "chief_complaint":     "Unable to generate summary — please review conversation history.",
            "symptom_summary":     ", ".join(features) if features else "No features recorded.",
            "ms_consistent_features": features,
            "mcdonald_assessment": {
                "dis_regions":  dis_regions,
                "dis_met":      len(dis_regions) >= 2,
                "dit_episodes": dit_episodes,
                "dit_met":      dit_episodes >= 2,
                "summary":      "McDonald criteria assessment could not be generated.",
            },
            "recommended_workup": ["Brain MRI with and without contrast", "Spinal cord MRI"],
            "confidence":         tier,
            "urgency":            "urgent" if tier in ("HIGH", "CRITICAL_EMERGENCY") else "soon" if tier == "MODERATE" else "routine",
            "neurologist_note":   "Please review full conversation transcript.",
        }
