"""
AxonIQ — Emergency Detection Tool
Pure regex, no LLM. Runs before everything else.
"""
from __future__ import annotations
import re
from typing import Optional

_PATTERNS = [
    (r"(sudden|woke up).{0,25}(face droop|arm weak|slurred speech|can.{0,5}t speak)",
     "Possible Acute Stroke",
     "Call emergency services immediately (999 / 911). Do NOT wait."),
    (r"(fever|high temp).{0,35}(neck).{0,25}(stiff|rigid|pain)",
     "Possible Meningitis",
     "Seek emergency care immediately. Fever + neck stiffness is a medical emergency."),
    (r"photophobia.{0,25}fever",
     "Possible Meningitis",
     "Seek emergency care immediately."),
    (r"(sudden|worst|thunderclap).{0,25}headache.{0,25}(ever|life|never|worst|severe)",
     "Possible Subarachnoid Haemorrhage",
     "Call emergency services immediately (999 / 911)."),
    (r"(both legs|feet|lower limbs).{0,35}(weak|paralyz|numb).{0,35}(spread|moving up|arms|hands)",
     "Possible Guillain-Barré Syndrome",
     "Seek emergency care immediately. Ascending weakness is a neurological emergency."),
    (r"ascending (paralysis|weakness|numbness)",
     "Possible Guillain-Barré Syndrome",
     "Seek emergency care immediately."),
    # v21: ported from NeuroCheck's EMERGENCY_FLAGS (gbs_suspected, 3rd
    # pattern) — the only one of NeuroCheck's emergency patterns not already
    # covered here. Catches ascending weakness described without explicitly
    # naming "both legs/feet" (e.g. "my weakness is spreading up toward my
    # arms"), which the pattern above requires.
    (r"weakness.{0,30}(spreading|moving up).{0,20}(arm|hand)",
     "Possible Guillain-Barré Syndrome",
     "Seek emergency care immediately. Ascending weakness is a neurological emergency."),
]


def check_emergency(text: str) -> Optional[dict]:
    """
    Returns emergency dict if text matches an emergency pattern, else None.
    dict: {label, action}
    """
    t = text.lower()
    for pattern, label, action in _PATTERNS:
        if re.search(pattern, t):
            return {"label": label, "action": action}
    return None
