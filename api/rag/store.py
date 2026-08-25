"""
AxonIQ — RAG Vector Store
ChromaDB backed. Indexes both MS clinical guidelines + 50 validated cases.
On HuggingFace Spaces, ChromaDB lives at /data/chroma_db (persistent volume).
"""
from __future__ import annotations
from typing import List
from loguru import logger

from api.core.config import CHROMA_DIR, DOCX_PATH, EMBED_MODEL, RAG_MIN_SCORE, RAG_TOP_K
from api.rag.knowledge import get_all_knowledge
from api.rag.extractor import chunk_documents

# Expected minimum chunk count — triggers rebuild if DB has fewer
_MIN_EXPECTED_CHUNKS = 55   # 10 guidelines + 50 cases + any docx chunks

_client     = None
_collection = None


def get_collection():
    global _client, _collection
    if _collection is not None:
        return _collection

    import chromadb
    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    ef = SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
    _client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    _collection = _client.get_or_create_collection(
        name="ms_knowledge",
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )
    return _collection


def build_knowledge_base(force: bool = False) -> None:
    """
    Populate ChromaDB from:
      1. knowledge.py  — 10 guidelines + 50 MS cases (always indexed)
      2. docx file     — additional document if present (optional)

    Idempotent unless force=True or chunk count is below expected minimum.
    """
    col = get_collection()
    current_count = col.count()

    if current_count >= _MIN_EXPECTED_CHUNKS and not force:
        logger.info("[RAG] Knowledge base already built ({} chunks). Skipping.", current_count)
        return

    if current_count > 0 and current_count < _MIN_EXPECTED_CHUNKS:
        logger.warning(
            "[RAG] Only {} chunks found (expected ≥{}). Rebuilding.",
            current_count, _MIN_EXPECTED_CHUNKS
        )
        force = True

    logger.info("[RAG] Building knowledge base...")

    # Source 1: guidelines + 50 cases from knowledge.py
    all_docs = get_all_knowledge()

    # Source 2: docx file (optional — does not fail if missing)
    from api.rag.extractor import extract_docx
    try:
        docx_docs = extract_docx(DOCX_PATH)
        if docx_docs:
            logger.info("[RAG] Loaded {} paragraphs from docx.", len(docx_docs))
            all_docs = all_docs + chunk_documents(docx_docs)
    except Exception as e:
        logger.warning("[RAG] Docx not loaded ({}). Continuing with knowledge.py only.", e)

    # Deduplicate by text content
    seen, deduped = set(), []
    for d in all_docs:
        key = d["text"][:80]
        if key not in seen:
            seen.add(key)
            deduped.append(d)

    ids   = [f"doc_{i}" for i in range(len(deduped))]
    texts = [c["text"] for c in deduped]
    metas = [c["meta"] for c in deduped]

    if force and current_count > 0:
        # Delete and recreate collection for clean rebuild
        _client.delete_collection("ms_knowledge")
        from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
        ef = SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
        global _collection
        _collection = _client.create_collection(
            name="ms_knowledge",
            embedding_function=ef,
            metadata={"hnsw:space": "cosine"},
        )

    col = get_collection()
    for i in range(0, len(texts), 50):
        col.upsert(
            ids=ids[i:i + 50],
            documents=texts[i:i + 50],
            metadatas=metas[i:i + 50],
        )

    logger.info("[RAG] Knowledge base ready: {} chunks indexed.", col.count())


def query(text: str, n_results: int = RAG_TOP_K) -> List[dict]:
    col = get_collection()
    if col.count() == 0:
        return []
    results = col.query(
        query_texts=[text],
        n_results=min(n_results, col.count()),
    )
    output = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        score = round(1 - dist, 3)
        if score >= RAG_MIN_SCORE:
            output.append({"text": doc, "source": meta.get("source", ""), "score": score})
    return output
