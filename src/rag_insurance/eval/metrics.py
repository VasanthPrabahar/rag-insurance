"""Retrieval metrics, implemented from scratch.

Relevance is binary and defined by the golden set: a retrieved chunk is
relevant iff it comes from one of the item's relevant_doc_names AND contains
the item's relevant_hint (case- and whitespace-insensitive substring).

recall@k here is hit@k: 1.0 if any relevant chunk appears in the top k,
else 0.0. We don't know the total number of relevant chunks in the corpus,
so set-recall isn't computable; hit@k is the honest per-question version
and averages to "fraction of questions where retrieval found the evidence".
"""

from __future__ import annotations

from rag_insurance.retrieval.dense import RetrievedChunk


def _normalize(text: str) -> str:
    # Collapse whitespace so hints match across PDF line breaks.
    return " ".join(text.split()).lower()


def relevance_flags(
    chunks: list[RetrievedChunk],
    relevant_doc_names: list[str],
    relevant_hint: str | None,
) -> list[bool]:
    if relevant_hint is None or not relevant_doc_names:
        return [False] * len(chunks)
    hint = _normalize(relevant_hint)
    docs = set(relevant_doc_names)
    return [chunk.doc_name in docs and hint in _normalize(chunk.content) for chunk in chunks]


def recall_at_k(flags: list[bool], k: int) -> float:
    return 1.0 if any(flags[:k]) else 0.0


def precision_at_k(flags: list[bool], k: int) -> float:
    if k <= 0:
        return 0.0
    return sum(flags[:k]) / k


def mrr(flags: list[bool]) -> float:
    for i, flag in enumerate(flags):
        if flag:
            return 1.0 / (i + 1)
    return 0.0
