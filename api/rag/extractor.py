"""
AxonIQ — Document Text Extractor
Extracts text from .docx files and splits into overlapping chunks.
No dependency on ChromaDB or the rest of the RAG stack.
"""
from __future__ import annotations
import re
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List

from loguru import logger


def extract_docx(path: Path) -> List[dict]:
    """Extract paragraph text from a .docx file and group into 3-para blocks."""
    if not path.exists():
        logger.warning("[RAG] Docx not found at {}", path)
        return []

    with zipfile.ZipFile(str(path)) as z:
        xml_bytes = z.read("word/document.xml")

    root = ET.fromstring(xml_bytes)
    ns = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
    paras = []
    for para in root.iter(f"{ns}p"):
        line = "".join(node.text or "" for node in para.iter(f"{ns}t")).strip()
        if line:
            paras.append(line)

    docs = []
    for i in range(0, len(paras), 3):
        block = " ".join(paras[i:i + 3])
        docs.append({"text": block, "meta": {"source": "AGENTIC_AI_chatbots.docx"}})
    return docs


def chunk_documents(docs: List[dict], max_len: int = 400) -> List[dict]:
    """Split long texts into overlapping sentence-boundary chunks."""
    out = []
    for d in docs:
        text = d["text"]
        if len(text) <= max_len:
            out.append(d)
            continue
        sentences = re.split(r"(?<=[.!?])\s+", text)
        chunk, chunk_len = [], 0
        for s in sentences:
            if chunk_len + len(s) > max_len and chunk:
                out.append({"text": " ".join(chunk), "meta": d["meta"]})
                chunk, chunk_len = [], 0
            chunk.append(s)
            chunk_len += len(s)
        if chunk:
            out.append({"text": " ".join(chunk), "meta": d["meta"]})
    return out
