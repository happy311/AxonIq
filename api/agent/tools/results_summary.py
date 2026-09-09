"""
AxonIQ — Deterministic MRI Results Summary
============================================

WHY THIS FILE EXISTS (v18):
The active LLM is frequently the free HuggingFace fallback
(meta-llama/Llama-3.1-8B-Instruct), not the primary Claude model. In
production this small model was observed to fail at turning structured MRI
findings into patient-facing prose in two ways:

  1. On the very first "your results are in" message, it sometimes emitted a
     generic filler ("the analysis is currently running...") even though the
     full findings were already in the prompt.
  2. On a plain follow-up ("is report arrived", "do i have ms"), it echoed
     the _POST_MRI_GUIDANCE_GOAL instruction block's raw "Risk tier: {tier} /
     Confirmed features: {features} / ..." lines back almost verbatim
     (and even truncated, e.g. dropping the tier value entirely) instead of
     writing a real answer.

For a clinical decision-support tool, "did the patient actually get told
their results and risk level" cannot depend on a small free model's
instruction-following on a given day. So the core facts — lesion findings,
McDonald DIS/DIT status, and a direct risk-level statement — are composed
here in plain Python and used verbatim, every time, regardless of which LLM
is active. The LLM (whichever one is active) is still used for open-ended
follow-up conversation (post_mri_guidance), but no longer solely responsible
for correctly conveying the critical facts.
"""
from __future__ import annotations

REGION_PLAIN = {
    "periventricular":   "around the fluid-filled ventricles in the brain",
    "juxtacortical":     "just beneath the brain's outer surface (the cortex)",
    "deep_white_matter": "the deep white matter of the brain",
    "unclassified":      "other areas outside the standard white-matter zones",
    "cerebral":          "the cerebral hemispheres",
    "brainstem":         "the brainstem",
    "cerebellar":        "the cerebellum",
    "spinal_cord":       "the spinal cord",
    "optic_nerve":       "the optic nerve",
}

_RISK_STATEMENT = {
    "LOW": (
        "Based on these results, this does not currently look like a pattern "
        "typical of Multiple Sclerosis."
    ),
    "WATCH": (
        "Based on these results, there's a low likelihood of MS right now — a "
        "few findings worth keeping an eye on, but nothing strongly pointing "
        "to it yet."
    ),
    "MODERATE": (
        "Based on these results, there's a moderate likelihood of Multiple "
        "Sclerosis — some findings are consistent with it, though not all the "
        "criteria typically used for a diagnosis are met yet."
    ),
    "HIGH": (
        "Based on these results, there's a high likelihood of Multiple "
        "Sclerosis — the pattern seen is strongly consistent with it."
    ),
    "CRITICAL_EMERGENCY": (
        "Based on these results together with your symptoms, there's a high "
        "likelihood of Multiple Sclerosis, and this needs urgent attention."
    ),
}


def plain_region(loc: str) -> str:
    return REGION_PLAIN.get(loc, loc.replace("_", " "))


def risk_statement_for_tier(tier: str) -> str:
    return _RISK_STATEMENT.get(tier, _RISK_STATEMENT["MODERATE"])


def build_mri_results_message(
    tier: str,
    findings: dict,
    dis_regions: list[str] | None = None,
    dit_episodes: int = 0,
) -> str:
    """
    Full deterministic patient-facing message for "your MRI results are in".
    Every fact here is read directly from `findings` / `dis_regions` /
    `dit_episodes` — nothing is invented, nothing depends on LLM output.
    """
    dis_regions = dis_regions or []
    locs        = findings.get("lesion_locations", []) or []
    # dis_met is recomputed here from dis_regions (the actual ground truth
    # passed in above) rather than trusted from findings — defense in depth
    # on top of the recompute already done in mri_analyzer.analyse_mri_text,
    # since dis_regions can differ from findings["dis_regions"] (this is the
    # merged/state-level list, which may include regions from earlier scans).
    dis_met     = len(dis_regions) >= 2
    dit_met     = bool(findings.get("dit_met", False)) and bool(findings.get("enhancing_lesions", False))
    lst_total   = findings.get("lst_ai_total_lesions")
    lst_vol     = findings.get("lst_ai_total_volume_mm3")

    lines = ["Your MRI results are in."]

    if lst_total is not None:
        plain_locs = ", ".join(plain_region(l) for l in locs) if locs else "the scanned regions"
        lesion_word = "lesion" if lst_total == 1 else "lesions"
        lines.append(
            f"The scan identified {int(lst_total)} {lesion_word} "
            f"(about {float(lst_vol or 0):.0f} mm³ total) in {plain_locs}."
        )
    elif locs:
        lines.append(f"The report describes lesions in {', '.join(plain_region(l) for l in locs)}.")
    else:
        lines.append("The report did not describe clear lesion findings.")

    if dis_regions:
        plain_dis = ", ".join(plain_region(r) for r in dis_regions)
        lines.append(
            f"Taking the symptoms you described into account as well, "
            f"involvement has now been noted across: {plain_dis}."
        )

    if dis_met:
        lines.append(
            "This meets the \"dissemination in space\" criterion (lesions in "
            "2 or more separate areas of the central nervous system) — one of "
            "the two McDonald criteria used to help diagnose MS."
        )
    if dit_met:
        episode_note = (
            f" (evidence spans {dit_episodes} separate episodes)"
            if dit_episodes and dit_episodes > 1
            else ""
        )
        lines.append(
            "It also meets the \"dissemination in time\" criterion (evidence "
            "of lesions from different points in time) — the other main "
            f"McDonald criterion{episode_note}."
        )
    if not dis_met and not dit_met:
        lines.append(
            "Right now, the MRI alone doesn't fully meet the McDonald "
            "criteria (lesions in multiple areas AND evidence they happened "
            "at different times) typically used to diagnose MS."
        )

    lines.append(risk_statement_for_tier(tier))
    lines.append(
        "A neurologist will still need to confirm this with a full clinical "
        "evaluation — an MRI alone can't give a final diagnosis."
    )
    lines.append("This assessment is for clinical decision support only and is not a diagnosis.")

    return "\n\n".join(lines)


# ── Longitudinal progression tracking ───────────────────────────────────────
# Added so repeat scans can be compared against a patient's own history
# (saved to the mri_results table — see api/database.py:save_mri_result /
# get_latest_mri_result). Every fact here is read directly from the current
# `findings` dict or the previously-saved row — nothing is invented.

_TIER_ORDER = ["LOW", "WATCH", "MODERATE", "HIGH", "CRITICAL_EMERGENCY"]


def extract_lesion_metrics(findings: dict) -> tuple[int | None, float | None]:
    """
    Best-effort (count, volume_mm3) extraction shared by the DB save path and
    the progression note, so both always agree on the same numbers.
    Prefers the quantitative NIfTI ensemble output (`lst_ai_total_lesions` /
    `lst_ai_total_volume_mm3`) when present; falls back to the LLM-extracted
    `lesion_count` string from a pasted text report (e.g. "3" or "multiple",
    the latter of which can't be turned into a number and returns None).
    """
    count = findings.get("lst_ai_total_lesions")
    if count is None:
        raw = findings.get("lesion_count")
        try:
            count = int(str(raw).strip())
        except (TypeError, ValueError):
            count = None
    else:
        try:
            count = int(count)
        except (TypeError, ValueError):
            count = None

    volume = findings.get("lst_ai_total_volume_mm3")
    try:
        volume = float(volume) if volume is not None else None
    except (TypeError, ValueError):
        volume = None

    return count, volume


def build_progression_note(
    previous: dict | None,
    tier: str,
    lesion_count: int | None,
    lesion_volume: float | None,
) -> str:
    """
    Deterministic patient-facing note comparing this scan to the last MRI
    result saved for this patient (any session). `previous` is the dict
    returned by database.get_latest_mri_result() — or None on a patient's
    very first saved scan.
    """
    if not previous:
        return (
            "\n\n📌 This is the first MRI result saved to your record. It will "
            "be used as a baseline so any future scans can be compared against it "
            "to track how things change over time."
        )

    lines = ["\n\n📈 **Compared to your last saved scan**"]
    prev_date = str(previous.get("created_at") or "")[:10]
    if prev_date:
        lines[0] += f" (recorded {prev_date})"
    lines[0] += ":"

    prev_count = previous.get("lesion_count")
    if lesion_count is not None and prev_count is not None:
        diff = lesion_count - prev_count
        if diff > 0:
            lines.append(f"- Lesion count increased from {prev_count} to {lesion_count} (+{diff}).")
        elif diff < 0:
            lines.append(f"- Lesion count decreased from {prev_count} to {lesion_count} ({diff}).")
        else:
            lines.append(f"- Lesion count is unchanged at {lesion_count}.")

    prev_vol = previous.get("lesion_volume_mm3")
    if lesion_volume is not None and prev_vol is not None:
        vdiff = lesion_volume - prev_vol
        pct = f" ({vdiff / prev_vol * 100:+.0f}%)" if prev_vol else ""
        lines.append(
            f"- Total lesion volume changed from {prev_vol:.0f} mm³ to "
            f"{lesion_volume:.0f} mm³{pct}."
        )

    prev_tier = previous.get("tier", "LOW")
    if tier != prev_tier:
        try:
            worse = _TIER_ORDER.index(tier) > _TIER_ORDER.index(prev_tier)
            direction = "risen" if worse else "eased"
        except ValueError:
            direction = "changed"
        lines.append(f"- Risk classification has {direction} from {prev_tier} to {tier}.")

    if len(lines) == 1:
        lines.append("- No quantitative change was detected versus the last saved scan.")

    lines.append(
        "\nThis comparison is for your own tracking only — please review any "
        "changes with your neurologist rather than interpreting them alone."
    )
    return "\n".join(lines)
