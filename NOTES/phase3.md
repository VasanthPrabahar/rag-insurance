# Phase 3 — Retrieval quality notes

Every change below ran `rag eval --skip-llm` before and after; rows live in
`eval/RESULTS.md`. Two of five changes were reverted — that is the system
working, not failing.

## Step 1 — Embedder truncation A/B (kept: bge-base-en-v1.5)

MiniLM truncates at 256 tokens, so half of every 512-token chunk was
invisible to the Phase 1-2 embedder. Two fixes tested:
(a) 256/32 chunks aligned to MiniLM: recall@5 0.577 → 0.654, but MRR fell
to 0.428 — more questions found *some* evidence, at worse ranks (smaller
chunks split context, and 340 chunks give more near-duplicates).
(b) bge-base-en-v1.5 at 512/64 (768-dim, 512-token capacity, query
instruction prefix): recall@5 0.769, MRR 0.571 — wins both axes. Kept (b).
The lesson: "make the chunk fit the model" and "make the model fit the
chunk" are both valid; only measurement says which.

## Step 2 — Structure-aware chunking (REVERTED)

Split ISO policies on their real hierarchy (PART A-F, sections, individual
definitions, section_path metadata prepended to content). Result: recall@5
0.769 → 0.615, MRR 0.571 → 0.466. A sibling-packing variant (merge small
same-part sections up to 512 tokens) did no better (0.615 / 0.435).

Why it lost, best hypothesis: bge embeddings of big mixed windows benefit
from co-occurring context (an insuring agreement next to its exclusions
gives the query more to latch onto), and hint-based relevance means
fragmenting a window can strand the hint in a chunk whose remaining text no
longer matches the question. Structure was supposed to fix "definition split
from exclusions" (g05/UM), but bge at 512 had already fixed those — the
remaining misses weren't chunking failures.

The splitter is kept in `chunker.py` behind `STRUCTURE_CHUNKING = False`
for reproducibility. The chunking-structure *rationale* stands — hierarchy
matters for policies — but as implemented it traded away more than it
gained. Revisit only with a concrete failing case in hand.

## Step 3 — Hybrid BM25 + dense with RRF (kept)

Postgres full-text (tsvector/GIN, ts_rank_cd, websearch_to_tsquery) top-20
fused with dense top-20 by reciprocal rank fusion. recall@5 held 0.769,
MRR 0.571 → 0.615, precision up.

**Why BM25 catches what dense misses:** an embedding is a lossy semantic
summary — surface forms are deliberately blurred away. "Temporary
transportation expenses" (g01, a Phase 2 target) is exactly the phrase in
Part D, but a 768-dim vector of a rental-car question lands nearer to
consumer-guide prose *about* rentals than to contract text that happens to
contain the term. BM25 does the opposite: it only sees surface forms —
exact terms, form numbers, dollar amounts. g01 went from missed to rank 1.

**RRF math:** each retriever contributes 1/(K + rank) per chunk, K=60,
ranks summed across lists. No score normalization is needed — cosine
similarity and ts_rank_cd live on incomparable scales, but *ranks* are
always comparable. K=60 flattens the curve so rank 1 (1/61) and rank 5
(1/65) differ modestly: agreement between retrievers dominates any single
retriever's enthusiasm.

## Step 4 — Cross-encoder reranking (measured, default OFF)

cross-encoder/ms-marco-MiniLM-L-6-v2 over the fused top-20.

**Bi- vs cross-encoder:** a bi-encoder embeds query and chunk separately —
comparison is a dot product of two independently-computed summaries,
computable against a whole corpus in milliseconds. A cross-encoder feeds
"query [SEP] chunk" through one transformer pass, letting every query token
attend to every chunk token — far more precise, and far too slow for more
than a shortlist (~20 pairs ≈ 430ms added; measured mean retrieval latency
92ms → 526ms per query).

Result: fixed g19 (TX minimums — evidence-present-but-missed, a Phase 2
target) and g04, but dropped g14/g15/g30; net recall@5 0.769 → 0.731, MRR
0.615 → 0.488. Likely mechanism: the ms-marco model truncates pairs at 512
tokens, and with 512-token chunks the tail — where g14's "temporary
substitute" and g15's "becomes insolvent" evidence sits — is never scored.
Per protocol, net-negative → default off (`--rerank` to enable; kept for
the Phase 6 latency comparison). Revisit with a longer-context reranker
(e.g. bge-reranker-base) rather than by shrinking chunks around this model.

## Step 5 — HNSW (kept; win is pedagogical at this scale)

`USING hnsw (embedding vector_cosine_ops)`, defaults m=16,
ef_construction=64. Measured on the 30 golden queries: exact scan 0.72ms
mean, HNSW 0.47ms, top-5 sets identical 30/30, eval metrics unchanged.

**Honestly:** at 174 chunks this saves a quarter of a millisecond —
sequential scan over 174 vectors is nothing. The reason to add it now is
mechanics, not speed: HNSW builds a multi-layer navigable-small-world
graph; a query greedily descends from a sparse top layer to denser lower
layers, examining `ef_search` candidates (default 40) per layer instead of
every vector. `m` (16) is the edges per node — bigger m, better recall,
bigger index; `ef_construction` (64) is how carefully the graph is built;
`ef_search` trades recall for speed at query time and is the knob you tune
first when recall dips. With 174 chunks and ef_search=40, the "approximate"
search examines a large fraction of the corpus — which is why results are
identical. The tradeoff becomes real around 10^5-10^6 vectors.

## The g19 lesson — why both metric layers matter

g19's arc across two phases: Phase 2 baseline — refused (corpus gap),
judge wrongly scored the refusal correct. Corpus fixed — evidence present
but ranked ~20th; retrieval metrics said fail while the judge, seeing a
refusal, said "fine". Phase 3 hybrid — still missed at k=5. Rerank —
finally rank 1, but rerank regressed the overall suite, so the pipeline
that ships still misses g19's cb020 chunk (the worksheet chunk covers it).
No single metric tells this story: retrieval metrics catch what the judge
can't see (evidence never surfaced), the judge catches what retrieval
can't see (the answer built from surfaced evidence is wrong or leaked from
memory), and only both layers together locate *where* in the pipeline the
failure lives. This is the whole argument for eval-gating at two levels.
