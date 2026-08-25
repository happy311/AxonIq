"""
AxonIQ — Agent State

SYSTEM DESIGN: Single TypedDict flows through the LangGraph each turn.
Phase is the ONLY driver of behaviour — no string-matching on LLM output.

Phase transitions (persisted in DB, not derived from text):
  gathering → concluded → mri_requested → mri_received

Clinical fields added (v11):
  dis_regions      — McDonald DIS: CNS regions with confirmed demyelination
  dit_episodes     — McDonald DIT: count of prior episodes that resolved
  symptom_timeline — Timeline per symptom (duration, onset pattern, recurrence)

These are persisted in the DB each turn and flow back into state on the next turn,
so the goal setter can apply McDonald 2017 criteria without re-parsing history.
"""
from __future__ import annotations
from typing import List, Optional, TypedDict


class AgentState(TypedDict):
    # ── Conversation ──────────────────────────────────────────────────────────
    session_id:   str
    user_message: str
    history:      List[dict]        # [{role, content}, ...]
    human_turns:  int

    # ── Phase (single source of truth for behaviour) ──────────────────────────
    # Values: "gathering" | "concluded" | "mri_requested" | "mri_received"
    phase:        str

    # ── Goal (set by goal_setter node; read by llm_response node) ────────────
    # Values: "gathering" | "conclude" | "conclude_with_mri" | "request_mri"
    #         | "mri_received" | "emergency"
    goal:         str

    # ── Classification (internal only — never shown in UI) ───────────────────
    tier:         str               # LOW | WATCH | MODERATE | HIGH | CRITICAL_EMERGENCY
    features:     List[str]         # MS features identified so far

    # ── McDonald 2017 Criteria ────────────────────────────────────────────────
    dis_regions:      List[str]     # CNS regions: optic_nerve, spinal_cord, brainstem, cerebellar, cerebral
    dit_episodes:     int           # 0 = first ever, 1 = one prior resolved, 2+ = DIT confirmed

    # ── Symptom Timeline ──────────────────────────────────────────────────────
    symptom_timeline: List[dict]    # [{symptom, duration, onset, resolved_before}]

    # ── MRI (NIfTI pipeline) ─────────────────────────────────────────────────
    # nifti_paths holds {"flair": "/tmp/..."} when a FLAIR file has been queued
    # for analysis this turn; None otherwise. The MRI backend is FLAIR-only —
    # a "t1" key may still appear for backward compatibility but is unused.
    nifti_paths:        Optional[dict]   # {"flair": path} or None
    mri_report:        Optional[dict]    # returned document from MRI service
    mri_service_failed: bool             # True when POST succeeded but service was unreachable

    # ── RAG ───────────────────────────────────────────────────────────────────
    rag_context:  str

    # ── Output ───────────────────────────────────────────────────────────────
    response:     str
    next_phase:   str               # phase to store in DB after this turn
