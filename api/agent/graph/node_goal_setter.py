"""
AxonIQ — Goal Setter Node (v14 bugfix)

Merge of v11.3 + v12 + v14 bugfixes:

  [from v11.3] WATCH + confirmed features → conclude_with_mri (not plain conclude)
    Any patient with a WATCH tier AND at least one confirmed MS-pattern feature
    should receive an MRI request. Plain conclude was previously given, which
    omitted imaging for borderline cases — clinically unacceptable.

  [v12 base]   All other logic unchanged from v12.

  [v14 BUG FIX 1] FAKE MRI RESULTS:
    Previously, ANY message sent while phase==mri_requested (e.g. "hello",
    "I haven't got my MRI yet") was silently treated as an MRI submission.
    node_mri_analysis then ran Path B (text analysis) on that plain text,
    producing a fake LOW-tier result like "The MRI scans have been reviewed..."
    FIX: Only trigger mri_received when ACTUAL nifti_paths are present in state.
    Without real data, stay in mri_requested and re-prompt with request_mri.

  [v14 BUG FIX 2] MRI FILES NOT SENT TO SERVER AFTER FAKE ASSESSMENT:
    After a fake assessment moved phase to mri_received, any subsequent real
    MRI upload was routed to post_mri_guidance (not mri_analysis) because the
    phase check fired before the nifti_paths check.
    FIX: Add PRIORITY 2 — if nifti_paths present, always route to mri_analysis
    regardless of current phase (even mri_received).
"""
from __future__ import annotations
from api.agent.state import AgentState

_TIER_ORDER = ["LOW", "WATCH", "MODERATE", "HIGH", "CRITICAL_EMERGENCY"]

# ── MRI report detection ──────────────────────────────────────────────────────

_MRI_KEYWORDS = frozenset([
    "findings:", "impression:", "impression /", "t2/flair", "t2 flair",
    "periventricular", "juxtacortical", "infratentorial", "enhancing",
    "non-enhancing", "demyelinating", "gadolinium", "gadovist", "contrast",
    "hyperintense", "white matter lesion", "dawson", "spinal cord",
    "cervical spine", "thoracic spine", "radiologist", "imaging center",
    "mri report", "sagittal", "axial", "flair", "cerebellar peduncle",
    "lateral ventricle", "neuroradiology", "electronically signed",
    "board certified", "dissemination in space", "dissemination in time",
    "relapsing", "plaque", "lesion load", "t2 hyperintense",
])


def _is_mri_report(msg: str) -> bool:
    """True if message looks like a pasted radiology report (≥3 keywords + >150 chars)."""
    if len(msg) < 150:
        return False
    msg_lower = msg.lower()
    return sum(1 for kw in _MRI_KEYWORDS if kw in msg_lower) >= 3


# ── Recheck-via-case-id detection ─────────────────────────────────────────────
# [v15 BUG FIX, v17 REVISED] FABRICATED MRI RESULTS: when a patient pushed with
# phrases like "can't you get the result using the case id", the (especially
# the small HF fallback) LLM would comply and hallucinate a full fake report
# rather than admit it had no such tool.
#
# v17: the app DOES have a real way to do this — the ensemble server's
# GET /result/<case_id> endpoint, the same one _call_nifti_service polls.
# So instead of just refusing, these phrases now route to goal "mri_recheck",
# which does one real, single-shot fetch of THIS session's actual case_id
# (see node_mri._handle_recheck) and answers with whatever it genuinely
# says — done / still processing / failed. Grounded, never invented.
_RECHECK_PHRASES = [
    "case id", "case number", "using the case", "by the case",
    "check the server", "retrieve the result", "retrieve it",
    "look it up", "pull up the result", "get the result", "fetch the result",
    "check again", "check the mri again", "check my mri again",
    "check my results again", "recheck", "re-check", "check now",
    "any update", "check my mri", "check the results",
]


def _asks_to_recheck_mri(msg: str) -> bool:
    msg_lower = msg.lower()
    return any(p in msg_lower for p in _RECHECK_PHRASES)


# ── MRI upload intent detection ───────────────────────────────────────────────
# NOTE (v17 bug fix): "mri result" / "my scan result" used to be here, but
# they're substrings of ordinary status questions like "how are my mri
# results" — that falsely matched _wants_to_upload_mri() even while
# phase=="mri_received" with a real report already in hand, kicking the
# patient back into "please upload again" instead of answering from the
# report they already have. Removed; genuine upload intent is still covered
# by the phrases below (file/button/verb-based, not "result"-based).
_MRI_UPLOAD_INTENTS = [
    "analyse my mri", "analyze my mri",
    "can you analyse", "can you analyze",
    "upload my mri", "upload my scan", "upload my nifti",
    "submit my mri", "submit my scan",
    "i have an mri", "i have a scan", "i have mri",
    "want to upload", "want to submit",
    "my mri scan", "my brain mri", "my mri files",
    "nifti file", "nii.gz", ".nii",
    "can i upload", "how do i upload", "how to upload",
    "mri button",
    "i got my mri", "i have my mri", "i got my scan",
]


def _wants_to_upload_mri(msg: str) -> bool:
    """True when the user explicitly wants to share or upload their MRI scan."""
    msg_lower = msg.lower()
    return any(intent in msg_lower for intent in _MRI_UPLOAD_INTENTS)


# ── Clinical question detection ───────────────────────────────────────────────

_QUESTION_SIGNALS = [
    "do i", "can i", "will i", "am i", "is it", "is this", "what is",
    "should i", "how do", "why do", "when will", "possibility", "chance",
    "risk", "likely", "could it be", "could i have", "might i have",
    "do you think", "what are the", "what does", "explain", "tell me",
]


def _is_clinical_question(msg: str) -> bool:
    """
    True if the message is a clinical question about the patient's condition
    (not an MRI report and not just a greeting).
    """
    if len(msg) > 400:          # Long messages are likely reports
        return False
    if _is_mri_report(msg):     # Already handled separately
        return False
    msg_lower = msg.lower()
    ends_with_q = msg.strip().endswith("?")
    has_signal  = any(q in msg_lower for q in _QUESTION_SIGNALS)
    return ends_with_q or has_signal


def _tier_idx(tier: str) -> int:
    return _TIER_ORDER.index(tier) if tier in _TIER_ORDER else 0


def node_goal_setter(state: AgentState) -> dict:
    phase        = state.get("phase", "gathering")
    tier         = state.get("tier", "LOW")
    turns        = state.get("human_turns", 0)
    dis_regions  = state.get("dis_regions", [])
    dit_episodes = state.get("dit_episodes", 0)
    user_msg     = state.get("user_message", "")
    idx          = _tier_idx(tier)

    # ── PRIORITY 1: Pasted MRI report — fires regardless of phase ─────────────
    # Handles the case where user pastes a radiologist text report at any point.
    if _is_mri_report(user_msg):
        return {"goal": "mri_received", "next_phase": "mri_received"}

    # ── PRIORITY 2: NIfTI files uploaded — fires regardless of phase ──────────
    # BUG FIX (Bug 2): Previously, if phase was already "mri_received" (e.g. from
    # a prior fake or text assessment), actual NIfTI uploads were silently routed
    # to post_mri_guidance and the LST-AI service was NEVER called.
    # Fix: nifti_paths present in state is the definitive signal that real NIfTI
    # analysis is required. This must override ANY current phase.
    nifti_paths = state.get("nifti_paths")
    if nifti_paths and nifti_paths.get("flair"):
        return {"goal": "mri_received", "next_phase": "mri_received"}

    # ── PRIORITY 3: User explicitly asking to upload / analyse their MRI ───────
    # Catches "can you analyse my MRI?", "I want to upload my scan", etc.
    # Without this, the LLM received a GATHER prompt with no MRI context and
    # responded "I'm a text-based assistant and cannot receive files" — wrong.
    # Routing to request_mri gives the LLM the 🔬 button instructions instead.
    #
    # v19 BUG FIX: this used to unconditionally set next_phase="mri_requested"
    # "regardless of phase" — including when phase was ALREADY "mri_received"
    # with a full report in hand. A patient mentioning wanting to upload
    # another scan (e.g. "I have another mri from last year") would silently
    # regress the phase and lose post_mri_guidance on every later turn, even
    # though nothing was actually uploaded. A genuine new upload still always
    # wins via PRIORITY 2 (nifti_paths present) regardless of phase, so this
    # priority only needs to touch phase when we're not already past it.
    if _wants_to_upload_mri(user_msg):
        next_phase = "mri_requested" if phase != "mri_received" else phase
        return {"goal": "request_mri", "next_phase": next_phase}

    # ── Phase: mri_requested ──────────────────────────────────────────────────
    if phase == "mri_requested":
        # "check the mri again" / "using the case id" / etc. → do a REAL
        # single-shot recheck against the ensemble server via node_mri
        # ._handle_recheck, instead of letting the LLM improvise an answer.
        if _asks_to_recheck_mri(user_msg):
            return {"goal": "mri_recheck", "next_phase": "mri_requested"}

        if _is_clinical_question(user_msg):
            # User is asking about their condition — re-run conclude_with_mri
            # so they get: symptoms named → consistent with MS → renewed MRI ask.
            # next_phase stays mri_requested so we still await the actual MRI.
            return {"goal": "conclude_with_mri", "next_phase": "mri_requested"}

        # BUG FIX (Bug 1): Previously ANY non-clinical message here triggered
        # "mri_received", which sent the user's casual text ("ok", "I'll get one")
        # to analyse_mri_text(). That returned empty findings and the LLM produced
        # a fake "MRI reviewed — risk is LOW" response with zero actual scan data.
        #
        # Fix: only Priority 1 (pasted report) or Priority 2 (NIfTI files) can
        # trigger mri_received. Everything else re-prompts the user to submit MRI.
        return {"goal": "request_mri", "next_phase": "mri_requested"}

    # ── Phase: mri_received — MRI processed; provide guidance, not more gathering ─
    # Returning to "gathering" after MRI analysis is clinically incoherent: the
    # assessment is complete and the patient needs guidance, not new symptom questions.
    if phase == "mri_received":
        return {"goal": "post_mri_guidance", "next_phase": "mri_received"}

    # ── Phase: concluded — backward-compat ────────────────────────────────────
    if phase == "concluded":
        return {"goal": "request_mri", "next_phase": "mri_requested"}

    # ── Phase: gathering — McDonald 2017 criteria fast-track ──────────────────

    # DIS: 2+ CNS regions | DIT: 1 region + 2+ prior episodes → HIGH immediately
    mcdonald_high = (
        len(dis_regions) >= 2
        or (len(dis_regions) >= 1 and dit_episodes >= 2)
    )
    if mcdonald_high and idx < _tier_idx("HIGH"):
        idx  = _tier_idx("HIGH")
        tier = "HIGH"

    # v19 BUG FIX: `tier` above was previously a LOCAL variable used only to
    # pick which goal to return — the McDonald-triggered HIGH upgrade was
    # never included in the returned dict, so state["tier"] stayed at its
    # old value. node_llm.py's conclude_with_mri prompt (and chat.py's
    # "urgent" vs "routine" urgency flag sent to the frontend) both read
    # tier straight from state, so a patient who clinically qualifies for
    # HIGH under McDonald DIS/DIT criteria could still get non-urgent
    # messaging. Every return below now includes "tier" so the (possibly
    # just-elevated) value actually persists.

    if idx >= _tier_idx("HIGH"):
        return {"goal": "conclude_with_mri", "next_phase": "mri_requested", "tier": tier}

    if idx >= _tier_idx("MODERATE") and turns >= 2:
        return {"goal": "conclude_with_mri", "next_phase": "mri_requested", "tier": tier}

    # [v11.3 fix] WATCH with any confirmed features → MRI is clinically warranted.
    # Vague-but-real neurological symptoms should not silently conclude without imaging.
    if idx >= _tier_idx("WATCH") and turns >= 5:
        features = state.get("features", [])
        if features:
            return {"goal": "conclude_with_mri", "next_phase": "mri_requested", "tier": tier}
        return {"goal": "conclude", "next_phase": "concluded", "tier": tier}

    if turns >= 6:
        return {"goal": "conclude", "next_phase": "concluded", "tier": tier}

    return {"goal": "gathering", "next_phase": "gathering", "tier": tier}
