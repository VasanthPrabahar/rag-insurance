# Phase notes — study guide

One line on what each phase actually teaches. Read in order; every claim
links to a measured run in `eval/RESULTS.md`.

- **[phase0.md](phase0.md)** — Corpus selection is a design act: why
  policies break naive RAG (hierarchy, defined terms, cross-references)
  and the ISO Parts A–D structure everything later depends on.
- **[phase1.md](phase1.md)** — Embeddings as lossy semantic compression,
  cosine similarity, why fixed windows break policy structure, and a
  truncation flaw shipped *knowingly* as the baseline to beat.
- **[phase1_failures.md](phase1_failures.md)** — Five hand-run questions
  that became the eval set's seed probes; register mismatch and
  split-definition failures identified before any metric existed.
- **[phase2.md](phase2.md)** — Golden-set design (hint AND doc = relevance),
  recall as the ceiling on answer quality, faithfulness vs correctness
  (the grounding-leak time bomb), and LLM-judge pitfalls demonstrated by
  a real judge error.
- **[phase3.md](phase3.md)** — Bi- vs cross-encoders, RRF math, why BM25
  catches what dense blurs, HNSW mechanics — and two "obviously right"
  changes (structure chunking, reranking) measured, losing, and reverted.
- **[phase4.md](phase4.md)** — Register mismatch and term expansion (HyDE's
  hand-rolled cousin), why citation verification must be mechanical (the
  fabricated-citations incident), refusal as a feature, citation pressure
  turning a synthesizer into a quoter, and the judge noise floor.
- **[phase5.md](phase5.md)** — Startup model loading (measured 30x tax),
  SSE with incremental JSON-field extraction, real latency budgets,
  idempotent delta ingestion, and the Airflow-out-of-compose judgment.
- **[phase6.md](phase6.md)** — Framework-last: LCEL as composition not
  capability, where a router agent genuinely pays (refusal short-circuit),
  and two apparent agent wins debunked as judge noise and prompt-cache.
- **[phase7.md](phase7.md)** — Golden sets are living artifacts: a relabel
  the router disagreed with (judgment baked into prompts), multi-part
  recall as hint coverage, and decompose measured as latency-without-recall
  on this corpus.
