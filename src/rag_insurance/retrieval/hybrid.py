"""Hybrid retrieval: dense + sparse candidates fused with RRF.

Reciprocal rank fusion: each ranked list contributes 1 / (K + rank) per
chunk (rank starts at 1, K=60). Chunks found by both retrievers accumulate
both contributions, so agreement floats to the top without any score
normalization across the two very different scoring scales.

This module is the single retrieval entry point; `search()` grows flags as
Phase 3 adds components (dense_only now, reranking in Step 4).
"""

from __future__ import annotations

import psycopg

from rag_insurance.retrieval import dense, sparse
from rag_insurance.retrieval.dense import RetrievedChunk

RRF_K = 60
CANDIDATES = 20


def rrf_fuse(lists: list[list[RetrievedChunk]], k_const: int = RRF_K) -> list[RetrievedChunk]:
    scores: dict[tuple[str, int], float] = {}
    chunks: dict[tuple[str, int], RetrievedChunk] = {}
    for ranked in lists:
        for rank, chunk in enumerate(ranked, start=1):
            key = (chunk.doc_name, chunk.chunk_index)
            scores[key] = scores.get(key, 0.0) + 1.0 / (k_const + rank)
            chunks.setdefault(key, chunk)

    fused = []
    for key, score in sorted(scores.items(), key=lambda kv: -kv[1]):
        chunk = chunks[key].model_copy(update={"score": round(score, 6)})
        fused.append(chunk)
    return fused


# Rerank default is OFF: measured in Phase 3, cross-encoder reranking fixed
# g04/g19 but dropped g14/g15/g30 (net recall@5 0.769 -> 0.731, MRR 0.615 ->
# 0.488) at ~5.7x retrieval latency — likely 512-token pair truncation
# hiding late-chunk evidence. See NOTES/phase3.md; toggle with --rerank.
def search(
    conn: psycopg.Connection,
    question: str,
    k: int = 5,
    dense_only: bool = False,
    use_rerank: bool = False,
    use_rewrite: bool = True,
    use_state_filter: bool = True,
) -> list[RetrievedChunk]:
    states = None
    if use_state_filter:
        from rag_insurance.retrieval.state_filter import allowed_states

        states = allowed_states(question)

    if dense_only:
        return dense.retrieve(conn, question, k=k, states=states)

    queries = [question]
    if use_rewrite:
        from rag_insurance.retrieval.rewrite import rewrite_query

        rewritten = rewrite_query(question)
        if rewritten:
            queries.append(rewritten)

    lists = []
    for query in queries:
        lists.append(dense.retrieve(conn, query, k=CANDIDATES, states=states))
        lists.append(sparse.retrieve(conn, query, k=CANDIDATES, states=states))
    fused = rrf_fuse(lists)
    if not use_rerank:
        return fused[:k]

    from rag_insurance.retrieval.rerank import rerank

    return rerank(question, fused[:CANDIDATES], k=k)
