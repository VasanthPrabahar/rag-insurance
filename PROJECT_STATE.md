# Project State

_Update this file at the end of every phase._

## Current phase

**v5 — Production service + ingestion pipeline** (complete)

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

### v3 — Retrieval quality (each step measured; see NOTES/phase3.md)
- KEPT: bge-base-en-v1.5 embeddings @ 512/64 chunks (recall@5 0.577 → 0.769,
  MRR 0.463 → 0.571) — beat the 256/32 MiniLM-aligned variant in an A/B
- KEPT: hybrid BM25 (tsvector/GIN) + dense fused with RRF k=60
  (MRR 0.571 → 0.615; g01 rental-expenses probe fixed by exact-term match)
- KEPT: HNSW index (identical results, 0.72 → 0.47 ms; pedagogical at 174
  chunks — honest tradeoff notes in NOTES/phase3.md)
- REVERTED: structure-aware chunking (recall 0.769 → 0.615; splitter kept
  behind STRUCTURE_CHUNKING=False)
- REVERTED to default-off: cross-encoder reranking (fixed g19/g04, broke
  g14/g15/g30, net negative, +430ms; toggle with --rerank)
- CLI toggles for Phase 6 ablations: --dense-only, --rerank/--no-rerank
- Retrieval latency now recorded per query in results JSONs

### v4 — Query intelligence + grounded generation (see NOTES/phase4.md)
- KEPT: query rewriting as additive term expansion (llama3.1:8b couldn't do
  full rewrites reliably — echoed few-shot examples). MRR 0.615 → 0.663, no
  regressions, +1.45s/query. g02 target NOT flipped: its evidence moved
  dense rank 28 → 14 but it's a chunk-representation problem (documented)
- KEPT: regex state filtering (named state + NAIC + ISO specimens; the
  specimens are state-agnostic by judgment call). Metrics flat, zero cost,
  guards cross-state contamination
- Grounded generation: structured JSON {answer, citations, refused},
  pydantic-validated with retry, MECHANICAL citation verification (drop
  hallucinated ids, force refusal on zero valid citations), CLI renders
  [doc_name > chunk] citations; verification counts recorded per eval run
- Spot-checks: deductible probe now grounds to a real cited chunk;
  TX-minimums extracts 30/60/25 perfectly when evidence is in context —
  no 8B generation ceiling; g19 failure is retrieval ranking only
- Ablation recorded: rewrite+rerank (0.731/0.506) — rerank pathology
  persists, stays default-off
- Citation prompt v1 caused extractive quoting (multi_hop faith 0.642);
  one measured prompt iteration fixed it (0.800 in gated recheck)
- v4-final: recall@5 0.769, MRR 0.578, faithfulness 0.860, correctness
  0.727 (correctness +0.054 vs v3; multi_hop 0.483 → 0.833; 29/29 citations
  mechanically verified, 0 fabricated). Faithfulness 0.043 under target,
  within the documented judge/rewrite noise floor — see NOTES/phase4.md

### v5 — Production service + ingestion pipeline (see NOTES/phase5.md)
- FastAPI service (`rag_insurance.api`): models load once at startup
  (measured: 6.63s load vs 0.21s warm embed), psycopg pool via lifespan;
  POST /ask streams SSE tokens (incremental answer-field extraction from
  the JSON generation stream — same evaluated path) + final event with
  verified citations and expand/retrieve/generate latency breakdown;
  /ingest, /health (DB+Ollama), /stats; structlog JSON per request
- Idempotent delta ingestion (`ingest/delta.py`): sha256 manifest table,
  delete-then-upsert per changed doc in one transaction; proven 15→0
  reprocessed across consecutive runs
- Airflow DAG (`dags/ingest_dag.py`), thin wrapper over the delta module;
  Airflow kept out of core deps/compose by documented judgment call
- Docker delivery: multi-stage uv Dockerfile, compose db+api, HF weights in
  a named volume, Ollama native-host via host.docker.internal
- Tests: 19 passing (API SSE with mocked Ollama, schema validation,
  citation-verification unit, chunker/metrics/roundtrip); CI needs no Ollama
- No retrieval changes; v5-final eval row confirms no regression

## Next steps (v6 — Agentic layer)

- LangChain/LangGraph router + query decomposition
- Honest latency comparison agentic vs direct pipeline (structlog data is
  in place)
- Open retrieval items still carried: g02 (targeted structure-chunking of
  insuring agreements), keyword-density noise (doc-type priors or
  bge-reranker), longer-context reranker revisit

## Key decisions

- **pgvector over a dedicated vector DB** — fewer moving parts to operate for
  a project this size; revisit if scale/query patterns demand it.
- **Native dev + Docker delivery** — iterate directly on the host; ship via
  Docker for reproducibility (pgvector already runs via docker-compose).
- **Ollama `llama3.1:8b`** for generation and judging — local-first; the
  weak-judge tradeoff is documented in `NOTES/phase2.md`.
- **bge-base-en-v1.5 embeddings** (768-dim, 512-token capacity) — won the
  Phase 3 A/B against chunk-shrinking for MiniLM; query-instruction prefix
  applied at query time.
- **Reverts are recorded, not erased** — structure chunking and reranking
  live in the codebase behind toggles with their eval rows preserved.
- **Eval-gated changes** — active as of v2: no retrieval/chunking/prompting
  change lands without a measured effect on the golden eval set.
- **Retrieval metrics are the primary gate** — deterministic; judge scores
  are a trend signal (weak judge, self-preference; see `NOTES/phase2.md`).
- **Naive-first baselines** — every "obvious" improvement is deferred until
  it can be measured against the naive pipeline.
