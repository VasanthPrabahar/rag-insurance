"""Cross-encoder reranking of fused candidates.

A cross-encoder reads query and chunk *together* through one transformer
pass, attending across both — far more accurate than comparing two
independently-produced embeddings, and far too slow to run over the whole
corpus. So: bi-encoder + BM25 propose candidates, cross-encoder re-orders
the top slice.
"""

from __future__ import annotations

from functools import lru_cache

from sentence_transformers import CrossEncoder

from rag_insurance.ingest.embedder import detect_device
from rag_insurance.retrieval.dense import RetrievedChunk

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@lru_cache(maxsize=1)
def get_model() -> CrossEncoder:
    return CrossEncoder(MODEL_NAME, device=detect_device())


def rerank(question: str, chunks: list[RetrievedChunk], k: int = 5) -> list[RetrievedChunk]:
    if not chunks:
        return []
    scores = get_model().predict([(question, chunk.content) for chunk in chunks])
    ranked = sorted(zip(chunks, scores, strict=True), key=lambda pair: -float(pair[1]))
    return [
        chunk.model_copy(update={"score": round(float(score), 4)}) for chunk, score in ranked[:k]
    ]
