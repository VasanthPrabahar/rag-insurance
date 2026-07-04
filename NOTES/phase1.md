# Phase 1 — Naive RAG notes

- **Embeddings as semantic compression**: all-MiniLM-L6-v2 maps a chunk of
  text to a single 384-dimensional vector — a lossy summary of "what this
  text is about." Two texts that discuss the same thing land near each other
  in that space even with zero word overlap ("hit a deer" ≈ "contact with an
  animal"), which is exactly what keyword search can't do. The loss matters
  too: one vector per 512-token chunk averages away specifics, so a chunk
  that mentions ten coverages is "about" all of them and matches none of
  them sharply.

- **Cosine similarity**: we compare vectors by the angle between them, not
  their length. Because we normalize embeddings at encode time (unit length),
  cosine similarity is just the dot product, and pgvector's `<=>` cosine
  distance is `1 - cos`. Scores in practice live in a narrow band (~0.3-0.7);
  the absolute value means little, only the ranking does — which is why
  retrieval metrics in Phase 2 are rank-based (recall@k, MRR).

- **Why fixed-size chunking breaks policy structure**: a 512-token window
  slides through the document blind to its shape. It splits an exclusion
  list across two chunks (retrieval finds exclusions 1-6 but not 7), severs
  a defined term from the definitions section that gives it meaning, and
  strands cross-references ("see Part D") in chunks that contain no Part D
  text. The retrieved chunk can be *about* the right topic and still be
  unusable for answering, because the operative sentence is in the
  neighboring chunk. This is the failure mode Phase 3's structure-aware
  chunking must measurably fix.

- **A truncation gotcha we accepted knowingly**: MiniLM's max sequence
  length is 256 tokens, but our chunks are 512 — so the embedding only
  "sees" the first half of every chunk. The back half of each chunk is
  retrievable only by luck. Left in place deliberately: it's part of the
  naive baseline, and fixing it (smaller chunks, or a longer-context
  embedder) is a measurable Phase 3 change.

- **Exact vs approximate search**: with no index, pgvector compares the
  query against all 390 chunk vectors — a sequential scan, perfect recall,
  and at this scale effectively instant. HNSW builds a navigable graph that
  makes search sublinear at the cost of approximate results (possible missed
  neighbors) and index build/maintenance time. At 390 vectors an index buys
  nothing; we add HNSW in Phase 3 only to *measure* the recall/latency
  tradeoff, which is worth understanding before the corpus grows.
