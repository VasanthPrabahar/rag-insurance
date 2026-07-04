# Project State

_Update this file at the end of every phase._

## Current phase

**v0 — Foundation** (complete)

## Completed

- uv project initialized (`rag_insurance`, src layout, Python 3.11+)
- Package skeleton: `src/rag_insurance/{ingest,retrieval,generation,eval}`
- Tooling: ruff (line-length 100), pre-commit (ruff + ruff-format) installed
- `scripts/download_data.py` — manifest-driven corpus downloader
- Corpus: 12 documents downloaded to `data/raw/` across TX, CA, AZ, and NAIC
  (see manifest in `scripts/download_data.py`; AZ DIFI direct access blocked
  by Cloudflare, sourced via Wayback Machine mirror — see `NOTES/phase0.md`)
- `eval/` directory created (empty, golden dataset comes in v6)
- Repo docs: README v1 (architecture diagram, phase table), LICENSE (MIT),
  `.env.example`, `.gitignore`

## Next steps (v1 — Ingestion)

- Parse downloaded PDFs while preserving document structure (headings,
  numbered parts/sections, defined-term markers)
- Build a chunker that respects policy hierarchy (Parts A-D, endorsements)
  rather than naive fixed-size splitting
- Decide on a parsing library (e.g. `pymupdf`/`unstructured`) and add it as
  a dependency only when v1 starts

## Key decisions

- **pgvector over a dedicated vector DB** — fewer moving parts to operate for
  a project this size; revisit if scale/query patterns demand it.
- **Native dev + Docker delivery** — iterate directly on the host; ship via
  Docker for reproducibility.
- **Ollama `llama3.1:8b`** for generation — local-first, no external API
  dependency during development.
- **Eval-gated changes** — starting in v6, no retrieval/chunking/prompting
  change lands without a measured effect on the golden eval set.
