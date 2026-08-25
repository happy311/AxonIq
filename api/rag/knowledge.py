"""
AxonIQ — RAG Knowledge Loader
Loads clinical knowledge from knowledge_data.json at runtime.
Data file lives next to this module.
Adding/editing knowledge = edit knowledge_data.json, no code changes needed.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import List

_DATA_FILE = Path(__file__).parent / "knowledge_data.json"


def get_all_knowledge() -> List[dict]:
    """Return all knowledge chunks for RAG indexing."""
    if not _DATA_FILE.exists():
        raise FileNotFoundError(
            f"Knowledge data file not found: {_DATA_FILE}\n"
            "Run scripts/export_knowledge.py to regenerate it."
        )
    return json.loads(_DATA_FILE.read_text(encoding="utf-8"))


def get_ms_guidelines() -> List[dict]:
    """Return only guideline chunks (category == 'guidelines')."""
    return [e for e in get_all_knowledge() if e.get("meta", {}).get("category") == "guidelines"]


def get_ms_cases() -> List[dict]:
    """Return only clinical case chunks."""
    return [e for e in get_all_knowledge() if e.get("meta", {}).get("category") != "guidelines"]
