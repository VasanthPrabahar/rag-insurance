# Project State

_Update this file at the end of every phase._

## Current phase

**v1 — Naive RAG end to end** (complete)

## Phase plan

Phase 1 is INTENTIONALLY naive — fixed-size chunking, dense-only retrieval,
no index, no citations — so later phases have a measured baseline to beat.
Structure-aware chunking, hybrid retrieval, and HNSW are Phase 3 measured
changes, gated by the Phase 2 eval harness.

| Phase | Scope |
|-------|-------|
| v0 | Foundation: scaffold, tooling, corpus downloader |
| v1 | Naive RAG end to end (fixed-size chunks, dense-only, grounded generation) |
| v2 | Eval harness: golden dataset + metrics; eval-gating starts |
| v3 | Measured retrieval upgrades: structure-aware chunking, hybrid + RRF, HNSW |
| v4 | Reranking + citations |
| v5 | Generation quality |
| v6 | Hardening |
| v7 | Packaging, Docker delivery |

## Completed

### v0 — Foundation
- uv project (`rag_insurance`, src layout, Python 3.11+), ruff + pre-commit
- Manifest-driven corpus downloader (`scripts/download_data.py`)
- Repo docs: README, LICENSE (MIT), `.env.example`, `.gitignore`

### v1 — Naive RAG end to end
- Corpus fix: added two full ISO Personal Auto Policy specimens
  (PP 00 01 06 98 via Nevada DOI, PP 00 01 09 18 via Virginia SCC) —
  14 documents total in `data/raw/`
- Manifest moved into the package (`rag_insurance.manifest`) as the single
  source of truth for downloader and parser
- Infra: `docker-compose.yml` running pgvector/pgvector:pg16
- Pipeline: `ingest/parser.py` (pymupdf/bs4 + header/footer stripping) →
  `ingest/chunker.py` (512 tokens, 64 overlap, MiniLM tokenizer,
  offset-sliced) → `ingest/embedder.py` (all-MiniLM-L6-v2, mps, normalized)
  → `ingest/store.py` (pgvector, idempotent upsert, exact search)
- `retrieval/dense.py` (top-k cosine via `<=>`), `generation/` (Ollama
  llama3.1:8b, answer-only-from-context prompt)
- CLI: `rag ingest`, `rag ask "..." --k 5` (typer entry point)
- Tests: chunker sizes/overlap unit test; store/retrieve roundtrip
  integration test (skips when DB is down)
- Ingested: 14 documents → 390 chunks
- Baseline failure analysis on 5 questions: `NOTES/phase1_failures.md`

## Next steps (v2 — Evaluation harness)

- Build a golden dataset in `eval/`: questions with expected source
  chunks/documents and reference answers, seeded by the Phase 1 failure
  analysis
- Retrieval metrics (recall@k, MRR) and answer metrics (groundedness,
  correctness) runnable as one command
- From v2 on, every retrieval/chunking/prompting change must show its eval
  delta before landing

## Key decisions

- **pgvector over a dedicated vector DB** — fewer moving parts to operate for
  a project this size; revisit if scale/query patterns demand it.
- **Native dev + Docker delivery** — iterate directly on the host; ship via
  Docker for reproducibility (pgvector already runs via docker-compose).
- **Ollama `llama3.1:8b`** for generation — local-first, no external API
  dependency during development.
- **Eval-gated changes** — starting in v2, no retrieval/chunking/prompting
  change lands without a measured effect on the golden eval set.
- **Naive-first baselines** — every "obvious" improvement (structure-aware
  chunking, hybrid retrieval, HNSW) is deferred until it can be measured
  against the naive Phase 1 pipeline.
