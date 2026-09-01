"""
AxonIQ — LLM Response Node (v13 merged)

SYSTEM DESIGN: LLM is a pure text generator. This node:
1. Builds the prompt from goal + context
2. Calls LLM (with retry on suspicious parse)
3. Parses classification data (tier, features, McDonald criteria, timeline)
4. Strips ALL classification text from prose before returning to UI

Parse strategy (in priority order):
  1. JSON block  — v11 format; most reliable; Claude always outputs this correctly
  2. TIER:/FOUND: tokens — legacy fallback for small HF models
  3. <ms_tier>/<ms_features> XML — oldest fallback

Retry logic: if tier regresses suspiciously on a non-first turn with existing
features, the node retries the LLM call once before accepting the result.

v13 merged changes:
  [from v12]   mri_service_failed short-circuit — bypasses LLM entirely when
               the MRI upload service was unreachable, so the patient sees the
               human-readable retry message instead of a confused LLM answer.

  [new fix]    Prose-only goals (conclude, conclude_with_mri, request_mri,
               mri_received) now call _prose_only() instead of _parse().
               These goals instruct the LLM not to output a JSON block, but
               some smaller HF models leak one anyway. Calling _parse() on
               that leaked block could corrupt tier/features/dis_regions with
               stale or empty values. The ratchet protects tier, and union logic
               protects features, but the safest fix is to skip extraction
               entirely for goals that never need it.
"""
from __future__ import annotations
import json
import re
from typing import Optional
from loguru import logger

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from api.agent.state import AgentState


def _safe_lesion_count(mri_report: dict, lst_total) -> int:
    """
    lst_total (from the ensemble server) is always numeric when present.
    mri_report["lesion_count"] comes from the free-text LLM extractor, whose
    own prompt explicitly allows "number or 'multiple'" — int() on that raw
    value crashes the graph (see: invalid literal for int() with base 10:
    'multiple'). Extract the first digit run instead, defaulting to 0.
    """
    if lst_total is not None:
        return int(lst_total)
    raw = mri_report.get("lesion_count", 0)
    if isinstance(raw, int):
        return raw
    match = re.search(r"\d+", str(raw))
    return int(match.group()) if match else 0
from api.agent.prompts import (
    SYSTEM_PROMPT, EMERGENCY_RESPONSE,
    gather, conclude, conclude_with_mri, request_mri, mri_received, rag_block, post_mri_guidance,
)

# ── Constants ─────────────────────────────────────────────────────────────────

_TIER_ORDER = ["LOW", "WATCH", "MODERATE", "HIGH", "CRITICAL_EMERGENCY"]
MAX_RETRIES = 2

# Goals that generate only prose — they must not update clinical state even if
# the LLM leaks a JSON classification block (which some small HF models do).
_PROSE_ONLY_GOALS = {"conclude", "conclude_with_mri", "request_mri", "mri_received", "post_mri_guidance"}

# ── Deterministic clinical tier floor ────────────────────────────────────────
# Applied after _parse(). Certain single MS findings warrant HIGH regardless
# of what the LLM emitted, because the tier rules text alone is not reliable
# enough for these high-stakes individual features.

_UHTHOFF_KEYWORDS = frozenset({
    "uhthoff", "heat sensitivity", "heat-sensitive", "worse in heat",
    "worse after hot", "worse after exercise", "exercise worsening",
    "temperature sensitivity", "heat worsens",
})


def _ratchet_tier(current: str, floor: str) -> str:
    cur_idx   = _TIER_ORDER.index(current) if current in _TIER_ORDER else 0
    floor_idx = _TIER_ORDER.index(floor)   if floor   in _TIER_ORDER else 0
    return _TIER_ORDER[max(cur_idx, floor_idx)]


def _apply_clinical_floor(tier: str, features: list[str], dis_regions: list[str]) -> str:
    """
    Post-parse deterministic tier floor for clinically unambiguous single findings.

    Rules:
    1. Optic neuritis (monocular vision loss + eye pain +/- colour fading) alone -> HIGH
    2. Lhermitte's sign (electric shock down spine on neck flexion) alone -> HIGH
    3. Uhthoff's phenomenon + any other confirmed MS feature -> HIGH
       (heat sensitivity alone is non-specific; combined with any MS sign it
        confirms prior CNS demyelination)
    """
    if tier in ("HIGH", "CRITICAL_EMERGENCY"):
        return tier  # already at or above floor; nothing to do

    feat_str = " ".join(f.lower() for f in features)

    # Rule 1: Optic neuritis -> HIGH
    optic_neuritis = (
        "optic neuritis" in feat_str
        or "optic_nerve" in dis_regions
        or ("monocular" in feat_str and ("eye pain" in feat_str or "painful" in feat_str))
        or ("vision loss" in feat_str and "eye pain" in feat_str)
        or ("colour" in feat_str and "faded" in feat_str and "eye" in feat_str)
        or ("color" in feat_str and "faded" in feat_str and "eye" in feat_str)
    )
    if optic_neuritis:
        logger.info("[LLM] Clinical floor: optic neuritis -> HIGH (was {})", tier)
        return _ratchet_tier(tier, "HIGH")

    # Rule 2: Lhermitte's sign -> HIGH
    lhermitte = (
        "lhermitte" in feat_str
        or ("electric shock" in feat_str and "spine" in feat_str)
        or ("electric shock" in feat_str and "neck" in feat_str)
        or ("electric shock" in feat_str and "back" in feat_str)
        or ("buzzing" in feat_str and "spine" in feat_str)
        or ("shock" in feat_str and "neck flex" in feat_str)
    )
    if lhermitte:
        logger.info("[LLM] Clinical floor: Lhermitte's sign -> HIGH (was {})", tier)
        return _ratchet_tier(tier, "HIGH")

    # Rule 3: Uhthoff's + any other confirmed MS feature -> HIGH
    has_uhthoff = any(kw in feat_str for kw in _UHTHOFF_KEYWORDS)
    if has_uhthoff:
        non_uhthoff = [f for f in features if not any(kw in f.lower() for kw in _UHTHOFF_KEYWORDS)]
        if non_uhthoff:
            logger.info("[LLM] Clinical floor: Uhthoff's + other MS features -> HIGH (was {})", tier)
            return _ratchet_tier(tier, "HIGH")

    return tier


# ── Regex: junk from HF inference API ────────────────────────────────────────

_JUNK = re.compile(
    r'\{[^{}]*"type"\s*:\s*"function"[^{}]*\}|<tool_call>[\s\S]*?</tool_call>|```(?:json)?[\s\S]*?```',
    re.DOTALL,
)

# ── Regex: legacy XML tag format ──────────────────────────────────────────────

_TAG = re.compile(
    r"<(ms_tier|ms_features)>(.*?)</\1>",
    re.DOTALL | re.IGNORECASE,
)

# ── Leaked instruction patterns (for small models that echo instructions) ─────

_LEAK = [
    re.compile(r'\[GOAL[^\]]*\]\s*', re.IGNORECASE),
    re.compile(r'Classify the symptoms gathered.*?(?=\n\n|\Z)', re.DOTALL | re.IGNORECASE),
    re.compile(
        r'(?:CRITICAL_EMERGENCY|HIGH|MODERATE|WATCH|LOW)\s*[=—–-]\s*(?:\d|\w).*?(?=\n|$)',
        re.IGNORECASE,
    ),
    re.compile(r'MS symptoms (?:confirmed|found) so far:.*?(?=\n|$)', re.IGNORECASE),
    re.compile(r'Confirmed so far:.*?(?=\n|$)', re.IGNORECASE),
    re.compile(r'\s*\bTIER:\s*\w+(?:\s+FOUND:\s*[^\n]*)?\s*$', re.MULTILINE | re.IGNORECASE),
    re.compile(r'^\s*\bFOUND:\s*[^\n]*$', re.MULTILINE | re.IGNORECASE),
    re.compile(r'REPLACE_WITH_\w+', re.IGNORECASE),
    re.compile(r'^Step\s+\d+\s*[—–-].*?(?=\n|$)', re.MULTILINE | re.IGNORECASE),
    re.compile(r'\[END OF RESPONSE\]', re.IGNORECASE),
    re.compile(r'Tier rules?:.*?(?=\n\n|\Z)', re.DOTALL | re.IGNORECASE),
    # v11: Also strip any remaining JSON-like classification block from prose
    re.compile(r'^\s*\{.*?"tier"\s*:.*?\}\s*$', re.MULTILINE | re.DOTALL | re.IGNORECASE),
]


def _clean_leaked_instructions(prose: str) -> str:
    for pattern in _LEAK:
        prose = pattern.sub("", prose)
    prose = re.sub(r"\n{3,}", "\n\n", prose)
    return prose.strip()


# ── JSON block extractor ──────────────────────────────────────────────────────

def _extract_json_block(raw: str) -> Optional[dict]:
    """
    Find the last valid JSON object in raw that contains a 'tier' key.
    Searches from the end so it finds our appended block, not inline JSON.
    Handles nested JSON (timeline entries are nested objects).
    """
    positions = [i for i, c in enumerate(raw) if c == "{"]
    for pos in reversed(positions):
        try:
            obj, _ = json.JSONDecoder().raw_decode(raw[pos:])
            if isinstance(obj, dict) and "tier" in obj:
                return obj
        except (json.JSONDecodeError, ValueError):
            continue
    return None


def _strip_json_block(text: str) -> str:
    """Remove the trailing JSON classification block from prose text."""
    last_brace = text.rfind("{")
    if last_brace == -1:
        return text
    tail = text[last_brace:]
    # Only strip if this looks like our classification block
    if '"tier"' in tail:
        return text[:last_brace].rstrip()
    return text


# ── Prose-only cleaner (for goals that must never update clinical state) ───────

def _prose_only(raw: str) -> str:
    """
    Strip all classification artifacts from raw LLM output without extracting
    any clinical values.  Used for goals where the LLM should produce prose
    only and must not update tier / features / dis_regions / dit_episodes even
    if it leaks a JSON block.
    """
    prose = _TAG.sub("", raw)
    prose = _strip_json_block(prose)
    prose = _clean_leaked_instructions(prose)
    prose = re.sub(r"\n{3,}", "\n\n", prose).strip()
    return prose


# ── Master parser (for gathering goal only) ───────────────────────────────────

def _parse(raw: str) -> tuple[str, str, list[str], list[str], int, list[dict]]:
    """
    Extract all classification data from raw LLM output.

    Returns:
        (prose, tier, features, dis_regions, dit_episodes, timeline)

    Only called for the 'gathering' goal. All other goals use _prose_only().
    """
    tier:         str        = "LOW"
    features:     list[str]  = []
    dis_regions:  list[str]  = []
    dit_episodes: int        = 0
    timeline:     list[dict] = []

    # ── Strategy 1: JSON block (v11 format) ───────────────────────────────────
    json_obj = _extract_json_block(raw)
    if json_obj:
        t = str(json_obj.get("tier", "LOW")).upper().strip()
        if t in _TIER_ORDER:
            tier = t

        raw_found = json_obj.get("found", [])
        if isinstance(raw_found, list):
            features = [str(f).strip() for f in raw_found if f]
        elif isinstance(raw_found, str) and raw_found.lower() not in ("none", ""):
            features = [f.strip() for f in raw_found.split(",") if f.strip()]

        raw_dis = json_obj.get("dis_regions", [])
        if isinstance(raw_dis, list):
            valid_regions = {"optic_nerve", "spinal_cord", "brainstem", "cerebellar", "cerebral"}
            dis_regions = [r for r in raw_dis if r in valid_regions]

        try:
            dit_episodes = int(json_obj.get("dit_episodes", 0))
        except (TypeError, ValueError):
            dit_episodes = 0

        raw_timeline = json_obj.get("timeline", [])
        if isinstance(raw_timeline, list):
            timeline = [t for t in raw_timeline if isinstance(t, dict) and "symptom" in t]

    # ── Strategy 2: TIER:/FOUND: tokens (legacy small-model format) ──────────
    if tier == "LOW" and not features:
        tier_match  = re.search(r'\bTIER:\s*(\w+)', raw, re.IGNORECASE)
        found_match = re.search(r'\bFOUND:\s*([^\n]+)', raw, re.IGNORECASE)

        if tier_match:
            t = tier_match.group(1).upper().strip()
            if t in _TIER_ORDER:
                tier = t

        if found_match:
            raw_feat = found_match.group(1).strip()
            raw_feat = re.sub(r'\s*\bTIER:.*$', '', raw_feat, flags=re.IGNORECASE).strip()
            if raw_feat.lower() not in ("none", "replace_with_features_or_none", ""):
                features = [
                    f.strip() for f in raw_feat.split(",")
                    if f.strip() and f.strip().lower() != "none"
                ]

    # ── Strategy 3: XML tags (oldest legacy format) ───────────────────────────
    if tier == "LOW" and not features:
        tags = {m.group(1).lower(): m.group(2).strip() for m in _TAG.finditer(raw)}
        if "ms_tier" in tags:
            t = tags["ms_tier"].upper()
            if t in _TIER_ORDER:
                tier = t
        if "ms_features" in tags:
            raw_feat = tags["ms_features"]
            if raw_feat.lower() not in ("none", "replace_with_features_or_none", "list", ""):
                features = [
                    f.strip() for f in raw_feat.split(",")
                    if f.strip() and f.strip().lower() != "none"
                ]

    # ── Build clean prose ─────────────────────────────────────────────────────
    prose = _TAG.sub("", raw)
    prose = _strip_json_block(prose)
    prose = _clean_leaked_instructions(prose)
    prose = re.sub(r"\n{3,}", "\n\n", prose).strip()

    return prose, tier, features, dis_regions, dit_episodes, timeline


# ── Main node ─────────────────────────────────────────────────────────────────

def node_llm_response(state: AgentState) -> dict:
    from api.agent.llm import llm

    goal             = state.get("goal", "gathering")
    history          = state.get("history", [])
    rag_ctx          = state.get("rag_context", "")
    mri_report       = state.get("mri_report")
    turns            = state.get("human_turns", 0)
    tier             = state.get("tier", "LOW")
    features         = state.get("features", [])
    dis_regions      = state.get("dis_regions", [])
    dit_episodes     = state.get("dit_episodes", 0)
    symptom_timeline = state.get("symptom_timeline", [])

    # ── [v15] Short-circuit: patient asked for a case-ID/server retrieval
    # shortcut. Answered with a fixed honest message instead of letting the
    # LLM improvise — see node_goal_setter._asks_for_retrieval_shortcut.
    if goal == "mri_no_shortcut":
        return {
            "response": (
                "I'm not able to look up or re-check results using a case ID — "
                "I only see results once the scan analysis finishes or you paste "
                "your radiologist's written report. It's still processing right "
                "now (this can take 10–20 minutes). Feel free to check back in a "
                "bit, or paste the written report in the meantime if you have one."
            ),
            "next_phase": "mri_requested",
        }

    # ── [v12] Short-circuit: MRI service was unreachable this turn ────────────
    # node_mri_analysis set mri_service_failed=True and pre-built the response.
    # Bypass the LLM entirely so the patient sees the retry message — not a
    # confused "no structured analysis available" LLM answer.
    if state.get("mri_service_failed"):
        preset = state.get("response", "")
        if preset:
            return {
                "response":   preset,
                "next_phase": state.get("next_phase", "mri_requested"),
            }

    # ── Build system prompt ───────────────────────────────────────────────────
    if goal == "emergency":
        system = EMERGENCY_RESPONSE
    else:
        if goal == "gathering":
            goal_instr = gather(turns, features, dis_regions)
        elif goal == "conclude_with_mri":
            goal_instr = conclude_with_mri(turns, tier, features, dis_regions, dit_episodes)
        elif goal == "conclude":
            goal_instr = conclude(turns, tier, features)
        elif goal == "request_mri":
            goal_instr = request_mri(turns, tier)
        elif goal == "mri_received":
            if mri_report:
                # Build a rich analysis string that includes LST-AI quantitative
                # data when the report came from an NIfTI scan
                base_summary = mri_report.get("summary", "")
                lst_total    = mri_report.get("lst_ai_total_lesions")
                lst_vol      = mri_report.get("lst_ai_total_volume_mm3")
                lst_regions  = mri_report.get("lst_ai_regions", [])
                lst_text     = mri_report.get("lst_ai_report_text", "")

                if lst_total is not None:
                    region_lines = "; ".join(
                        f"{r.get('region','?')}: {r.get('num_lesions',0)} lesions "
                        f"({r.get('lesion_volume_mm3',0):.2f} mm³)"
                        for r in lst_regions
                    ) or "no regional breakdown"
                    quant_summary = (
                        f"Automated MRI segmentation found {lst_total} lesions "
                        f"(total volume {lst_vol:.2f} mm³). "
                        f"Regional breakdown: {region_lines}."
                    )
                    analysis = f"{quant_summary}\n\n{base_summary}".strip()
                    if lst_text:
                        analysis += f"\n\nFull MRI report text:\n{lst_text[:2000]}"
                else:
                    analysis = base_summary or str(mri_report)

                goal_instr = mri_received(
                    turns=turns,
                    mri_analysis=analysis or "No structured analysis available.",
                    dis_met=bool(mri_report.get("dis_met", False)),
                    dit_met=bool(mri_report.get("dit_met", False)),
                    enhancing_lesions=bool(mri_report.get("enhancing_lesions", False)),
                    lesion_count=_safe_lesion_count(mri_report, lst_total),
                    lesion_locations=mri_report.get("lesion_locations", []),
                    conv_dis_regions=dis_regions,
                    conv_dit_episodes=dit_episodes,
                )
            else:
                goal_instr = mri_received(
                    turns=turns,
                    mri_analysis="No structured analysis available — interpret the raw report text the user provided.",
                    conv_dis_regions=dis_regions,
                    conv_dit_episodes=dit_episodes,
                )
        elif goal == "post_mri_guidance":
            goal_instr = post_mri_guidance(turns, tier, features, dis_regions, dit_episodes)
        else:
            goal_instr = gather(turns, features, dis_regions)

        system = SYSTEM_PROMPT + goal_instr + rag_block(rag_ctx)

    # ── Build message chain ───────────────────────────────────────────────────
    msgs = [SystemMessage(content=system)]
    for m in history[:-1]:
        msgs.append(
            HumanMessage(content=m["content"]) if m["role"] == "human"
            else AIMessage(content=m["content"])
        )
    msgs.append(HumanMessage(content=state["user_message"]))

    # ── LLM call with retry on suspicious parse ───────────────────────────────
    prose        = ""
    new_tier     = tier
    new_features = features
    new_dis      = dis_regions
    new_dit      = dit_episodes
    new_timeline = symptom_timeline

    for attempt in range(MAX_RETRIES + 1):
        try:
            raw = llm.invoke(msgs)
            raw = raw.content if hasattr(raw, "content") else str(raw)
            raw = _JUNK.sub("", raw).strip()
        except Exception as e:
            logger.error("[LLM] Error on attempt {}: {}", attempt, e)
            if attempt < MAX_RETRIES:
                logger.info("[LLM] Retrying... (attempt {})", attempt + 1)
                continue
            return {
                "response":   "I'm having trouble reaching the AI service right now. Please try again in a moment.",
                "next_phase": state.get("next_phase", "gathering"),
            }

        # [new fix] Prose-only goals: strip classification artifacts but do NOT
        # extract any clinical values — the LLM was told not to output JSON here,
        # and any leaked block must not corrupt tier / features / dis / dit.
        if goal in _PROSE_ONLY_GOALS:
            prose = _prose_only(raw)
            break  # No retry logic needed — we're not parsing state

        # Gathering goal: full parse + clinical floor + retry on suspicious regression
        prose, new_tier, new_features, new_dis, new_dit, new_timeline = _parse(raw)

        # Deterministic clinical floor: single optic neuritis, Lhermitte's, or
        # Uhthoff's + any MS feature must reach HIGH regardless of LLM tier.
        new_tier = _apply_clinical_floor(new_tier, new_features, new_dis)

        is_suspicious = (
            goal == "gathering"
            and attempt < MAX_RETRIES
            and turns > 1
            and features                 # we had features before
            and new_tier == "LOW"
            and not new_features         # LLM forgot everything
        )
        if is_suspicious:
            logger.warning("[LLM] Suspicious tier regression on attempt {} — retrying", attempt)
            continue

        break  # parse looks good

    # ── For prose-only goals, return without touching clinical state ───────────
    if goal in _PROSE_ONLY_GOALS:
        if not prose:
            prose = "Thank you — please let me know if you have any further questions."
        return {
            "response":   prose,
            "next_phase": state.get("next_phase", state.get("phase", "gathering")),
        }

    # ── Gathering: merge parsed values into state (all ratcheted / unioned) ───
    existing_idx = _TIER_ORDER.index(tier)     if tier     in _TIER_ORDER else 0
    new_idx      = _TIER_ORDER.index(new_tier) if new_tier in _TIER_ORDER else 0
    final_tier   = _TIER_ORDER[max(existing_idx, new_idx)]

    all_features = list(dict.fromkeys(features + new_features))
    all_dis      = list(dict.fromkeys(dis_regions + new_dis))
    final_dit    = max(dit_episodes, new_dit)

    # Apply floor once more on the fully merged feature set (union may have
    # combined signals that were split across turns).
    final_tier = _apply_clinical_floor(final_tier, all_features, all_dis)

    timeline_map = {e["symptom"]: e for e in symptom_timeline}
    for entry in new_timeline:
        timeline_map[entry["symptom"]] = entry
    final_timeline = list(timeline_map.values())

    logger.debug(
        "[LLM] Parsed tier={} features={} dis={} dit={} prose_len={}",
        new_tier, new_features, new_dis, new_dit, len(prose),
    )

    if not prose:
        prose = "Thank you for sharing that. Could you tell me more about when your symptoms first started?"

    return {
        "response":         prose,
        "tier":             final_tier,
        "features":         all_features,
        "dis_regions":      all_dis,
        "dit_episodes":     final_dit,
        "symptom_timeline": final_timeline,
    }
