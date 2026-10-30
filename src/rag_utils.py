"""Pure RAG helpers used by ingestion and the retrieval pipeline."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

from langchain_core.documents import Document

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def add_contextual_header(doc: Document) -> Document:
    """Contextual chunk headers: filename + page ride with the embedded text."""
    source = doc.metadata.get("source", "unknown")
    page = doc.metadata.get("page", "?")
    filename = Path(str(source)).name
    header = f"[Document: {filename} | Page: {page}]"
    body = doc.page_content.strip()
    if body.startswith("[Document:"):
        return doc
    headed = doc.model_copy()
    headed.page_content = f"{header}\n{body}"
    return headed


def document_key(doc: Document) -> tuple:
    source = str(doc.metadata.get("source", ""))
    page = doc.metadata.get("page", None)
    seq = doc.metadata.get("seq_index", None)
    if seq is None:
        return (source, page, doc.page_content[:80])
    return (source, page, seq)


def unique_documents(docs: Iterable[Document]) -> list[Document]:
    seen: set[tuple] = set()
    out: list[Document] = []
    for doc in docs:
        key = document_key(doc)
        if key in seen:
            continue
        seen.add(key)
        out.append(doc)
    return out


def unique_keep_order(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        key = item.strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def reciprocal_rank_fusion(
    ranked_lists: list[list[Document]],
    k: int = 60,
) -> list[Document]:
    """RRF fusion of several ranked retrieval lists (fusion retrieval)."""
    scores: dict[tuple, float] = {}
    first_doc: dict[tuple, Document] = {}
    for ranked in ranked_lists:
        for rank, doc in enumerate(ranked, start=1):
            key = document_key(doc)
            first_doc.setdefault(key, doc)
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
    ordered = sorted(scores, key=scores.get, reverse=True)
    return [first_doc[key] for key in ordered]


def expand_with_neighbors(
    hits: list[Document],
    corpus: list[Document],
    num_neighbors: int = 1,
) -> list[Document]:
    """Pad each hit with neighboring chunks from the same source file."""
    by_source: dict[str, dict[int, Document]] = {}
    for doc in corpus:
        source = str(doc.metadata.get("source", ""))
        seq = doc.metadata.get("seq_index")
        if seq is None:
            continue
        by_source.setdefault(source, {})[int(seq)] = doc

    expanded: list[Document] = []
    for hit in hits:
        source = str(hit.metadata.get("source", ""))
        seq = hit.metadata.get("seq_index")
        index = by_source.get(source, {})
        if seq is None or not index:
            expanded.append(hit)
            continue
        seq = int(seq)
        parts = []
        for i in range(seq - num_neighbors, seq + num_neighbors + 1):
            neighbor = index.get(i)
            if neighbor is not None:
                parts.append(neighbor.page_content)
        merged = hit.model_copy()
        merged.page_content = "\n\n".join(parts) if parts else hit.page_content
        expanded.append(merged)
    return unique_documents(expanded)


def compress_chunk(query: str, text: str, max_sentences: int = 6) -> str:
    """Keep sentences that overlap the query; fall back to the original text."""
    sentences = [s.strip() for s in _SENTENCE_SPLIT.split(text) if s.strip()]
    if len(sentences) <= 2:
        return text
    terms = {t.lower() for t in re.findall(r"[A-Za-z0-9]+", query) if len(t) > 2}
    if not terms:
        return text
    scored = []
    for sentence in sentences:
        tokens = {t.lower() for t in re.findall(r"[A-Za-z0-9]+", sentence)}
        overlap = len(terms & tokens)
        if overlap:
            scored.append((overlap, sentence))
    scored.sort(key=lambda item: item[0], reverse=True)
    kept = [sentence for _, sentence in scored[:max_sentences]]
    if not kept:
        return text
    ordered = [s for s in sentences if s in kept]
    return " ".join(ordered)
