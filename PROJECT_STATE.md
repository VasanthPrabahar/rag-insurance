# Project State

_Update this file at the end of every phase._

## Current phase

**v2 — Eval harness + corpus rebalance** (complete)

## Phase plan

Phase 1 was INTENTIONALLY naive — fixed-size chunking, dense-only retrieval,
no index, no citations — so later phases have a measured baseline to beat.
From v2 on, every retrieval/chunking/prompting change must show its eval
delta before landing.

| Phase | Scope |
|-------|-------|
| v0 | Foundation: scaffold, tooling, corpus downloader |
| v1 | Naive RAG end to end (fixed-size chunks, dense-only, grounded generation) |
| v2 | Eval harness + corpus rebalance: golden dataset, metrics, CI; eval-gating starts |
| v3 | Structure-aware chunking, hybrid BM25/dense with RRF, cross-encoder reranking, HNSW |
| v4 | Query rewriting, metadata filtering, citation-grounded answers with refusal |
| v5 | FastAPI service, Airflow ingestion DAG, full Docker delivery |
| v6 | LangChain/LangGraph agentic layer (router, decomposition), honest latency comparison |
| v7 | Polish, demo UI, final README |

## Completed

### v0 — Foundation
- uv project (`rag_insurance`, src layout, Python 3.11+), ruff + pre-commit
- Manifest-driven corpus downloader; repo docs, LICENSE (MIT)

### v1 — Naive RAG end to end
- Full pipeline: pymupdf/bs4 parser → fixed 512-token chunks (64 overlap) →
  MiniLM embeddings (mps) → pgvector (exact search) → dense top-k →
  Ollama llama3.1:8b with answer-only-from-context prompt
- CLI (`rag ingest`, `rag ask`), docker-compose pgvector, tests
- Baseline failure analysis on 5 questions (`NOTES/phase1_failures.md`)

### v2 — Eval harness + corpus rebalance
- `eval/golden_set.json`: 30 questions grounded in the actual corpus
  (12 lookup / 6 multi-hop / 6 state-specific / 4 out-of-scope /
  2 adversarial), including all five Phase 1 known-failure probes; every
  hint validated chunk-level against the real corpus
- `rag_insurance/eval/`: metrics from scratch (hit-style recall@k,
  precision@k, MRR; relevance = doc match AND hint substring),
  LLM-as-judge (faithfulness + correctness, JSON mode, one retry),
  runner writing `eval/results/{timestamp}_{tag}.json` + `eval/RESULTS.md`
- CLI: `rag eval --k 5 [--skip-llm] [--tag NAME]` with per-category table
- CI: GitHub Actions (ruff, pytest, retrieval-only eval against a pgvector
  service container, fresh ingest, HF model cached); badge in README
- Corpus rebalance (first eval-gated change): removed the AZ 2023 Auto
  Premium Report (233 chunks of rate-table noise, 60% of corpus); added
  TDI Automobile Insurance Guide cb020 (HTML; fills the TX 30/60/25 gap)
  and the AZ New Driver's Guide (Wayback mirror). 15 docs, ~174 chunks,
  largest doc ≈ 18%
- Measured result (`eval/RESULTS.md`): recall@5 0.500 → 0.538, MRR
  0.388 → 0.424 (v1-naive-dirty-corpus → v2-clean-corpus, same labels);
  0.577 / 0.463 after a documented g19 label correction. Faithfulness flat
  (0.893 → 0.897); state_specific recall drove the gain (0.500 → 0.667).
  Still failing and targeted for v3: g01 (rental/transportation expenses)
  and g02 (friend/permissive use) — policy legalese loses to
  conversational phrasing in dense retrieval

## Next steps (v3 — Retrieval upgrades, all eval-gated)

- Structure-aware chunking (respect policy Parts/sections; stop splitting
  definitions from exclusions)
- Hybrid retrieval: BM25 + dense with reciprocal rank fusion
- Cross-encoder reranking of the fused candidate list
- HNSW index with measured recall/latency tradeoff vs exact scan
- Each lands separately with its eval delta in `eval/RESULTS.md`

## Key decisions

- **pgvector over a dedicated vector DB** — fewer moving parts to operate for
  a project this size; revisit if scale/query patterns demand it.
- **Native dev + Docker delivery** — iterate directly on the host; ship via
  Docker for reproducibility (pgvector already runs via docker-compose).
- **Ollama `llama3.1:8b`** for generation and judging — local-first; the
  weak-judge tradeoff is documented in `NOTES/phase2.md`.
- **Eval-gated changes** — active as of v2: no retrieval/chunking/prompting
  change lands without a measured effect on the golden eval set.
- **Retrieval metrics are the primary gate** — deterministic; judge scores
  are a trend signal (weak judge, self-preference; see `NOTES/phase2.md`).
- **Naive-first baselines** — every "obvious" improvement is deferred until
  it can be measured against the naive pipeline.
