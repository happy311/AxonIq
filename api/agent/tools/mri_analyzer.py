"""
AxonIQ — MRI Text Analyser (v11.1)

Extracts structured MS-relevant findings from a radiologist's free-text MRI report.
Returns a standardised dict regardless of report format.

v11.1 additions:
  - dis_regions: maps lesion locations to McDonald 2017 DIS regions
  - dit_met:     true when report shows simultaneous enhancing + non-enhancing lesions
  - Enhanced tier rules: HIGH for any enhancing lesion or 2+ lesion zones
  - Text limit raised from 3000 → 6000 chars to handle full reports
"""
from __future__ import annotations
from loguru import logger

_EXTRACT_PROMPT = """You are a neuroradiology AI. Extract structured MS-relevant findings from the MRI report.

Return ONLY a valid JSON object with EXACTLY these keys — no other text, no markdown fences:
{{
  "has_lesions": true/false,
  "lesion_locations": ["periventricular", "juxtacortical", "infratentorial", "spinal_cord", "cerebellar", "brainstem"],
  "lesion_count": "number or 'multiple'",
  "enhancing_lesions": true/false,
  "t2_flair_abnormal": true/false,
  "dis_met": true/false,
  "dit_met": true/false,
  "dis_regions": ["cerebral", "brainstem", "cerebellar", "spinal_cord", "optic_nerve"],
  "tier": "LOW/WATCH/MODERATE/HIGH",
  "summary": "1-2 sentence plain English summary of key findings"
}}

CLASSIFICATION RULES — apply strictly:
- tier HIGH   = lesions in 2+ distinct anatomical zones OR any enhancing (active) lesions OR clearly demyelinating pattern
- tier MODERATE = lesions in exactly 1 zone, no enhancement, non-specific
- tier WATCH  = borderline / non-specific white matter changes
- tier LOW    = normal MRI or no MS-relevant findings

- dis_met = true if lesions span 2 or more of: periventricular, juxtacortical, infratentorial, spinal cord
- dit_met = true if report shows BOTH enhancing (acute) AND non-enhancing (chronic) lesions simultaneously

DIS REGION MAPPING (use for dis_regions field):
  periventricular, juxtacortical, cortical → "cerebral"
  infratentorial → "brainstem" AND/OR "cerebellar" (add both if cerebellar peduncle/pons/medulla mentioned)
  cerebellar, cerebellar peduncle → "cerebellar"
  brainstem, pons, medulla, midbrain → "brainstem"
  spinal cord, cervical, thoracic, lumbar → "spinal_cord"
  optic nerve → "optic_nerve"

MRI REPORT:
{report}"""


def analyse_mri_text(report_text: str, llm) -> dict:
    """Parse a free-text radiology report into structured MS findings."""
    if not report_text or len(report_text.strip()) < 20:
        return _no_report()

    # Check if user is saying they don't have an MRI
    no_mri_signals = [
        "no mri", "don't have", "dont have", "haven't had", "havent had",
        "not done", "no scan", "no report", "not had",
    ]
    if any(s in report_text.lower() for s in no_mri_signals) and len(report_text) < 200:
        return _no_report()

    # Raise text limit from 3000 → 6000 to handle full radiology reports
    prompt = _EXTRACT_PROMPT.format(report=report_text[:6000])

    try:
        from langchain_core.messages import HumanMessage, SystemMessage
        import json, re

        resp = llm.invoke([
            SystemMessage(content="You extract structured data from MRI reports. Output ONLY valid JSON."),
            HumanMessage(content=prompt),
        ])
        raw = resp.content if hasattr(resp, "content") else str(resp)

        # Strip markdown fences
        raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
        data = json.loads(raw)

        # Validate and normalise tier
        valid_tiers = {"LOW", "WATCH", "MODERATE", "HIGH"}
        if data.get("tier") not in valid_tiers:
            data["tier"] = "LOW"

        # Ensure all expected keys are present (with safe defaults)
        data.setdefault("has_lesions",       False)
        data.setdefault("lesion_locations",  [])
        data.setdefault("lesion_count",      "0")
        data.setdefault("enhancing_lesions", False)
        data.setdefault("t2_flair_abnormal", False)
        data.setdefault("dis_met",           False)
        data.setdefault("dit_met",           False)
        data.setdefault("dis_regions",       [])
        data.setdefault("summary",           "")

        # Safety net: if there are enhancing lesions, tier must be at least HIGH
        if data["enhancing_lesions"] and data["tier"] not in ("HIGH",):
            data["tier"] = "HIGH"

        # Safety net: if DIS is met, tier must be at least HIGH
        if data["dis_met"] and data["tier"] not in ("HIGH",):
            data["tier"] = "HIGH"

        # Deduplicate and validate dis_regions
        valid_regions = {"cerebral", "brainstem", "cerebellar", "spinal_cord", "optic_nerve"}
        data["dis_regions"] = list(dict.fromkeys(
            r for r in data.get("dis_regions", []) if r in valid_regions
        ))

        logger.info(
            "[MRI Analyser] tier={} dis_met={} dit_met={} dis_regions={} enhancing={}",
            data["tier"], data["dis_met"], data["dit_met"],
            data["dis_regions"], data["enhancing_lesions"],
        )
        return data

    except Exception as e:
        logger.error("[MRI Analyser] Parse failed: {}", e)
        return {
            "has_lesions":       False,
            "lesion_locations":  [],
            "lesion_count":      "0",
            "enhancing_lesions": False,
            "t2_flair_abnormal": False,
            "dis_met":           False,
            "dit_met":           False,
            "dis_regions":       [],
            "tier":              "LOW",
            "summary":           "Could not parse MRI report — please provide the full radiologist report text.",
        }


def _no_report() -> dict:
    return {
        "has_lesions":       False,
        "lesion_locations":  [],
        "lesion_count":      "0",
        "enhancing_lesions": False,
        "t2_flair_abnormal": False,
        "dis_met":           False,
        "dit_met":           False,
        "dis_regions":       [],
        "tier":              "LOW",
        "summary":           "No MRI report provided.",
    }


def mri_tier_to_score(tier: str) -> int:
    return {"LOW": 5, "WATCH": 30, "MODERATE": 55, "HIGH": 80}.get(tier, 5)
