# Auto Insurance RAG — built from scratch, measured at every step

## Vision

Insurance policies are exactly the kind of document naive RAG breaks on: deeply
nested clauses, defined terms that only mean something in context, exclusions
that override exceptions to exclusions, and cross-references that span pages
or whole sections. This project builds a retrieval-augmented generation system
for auto insurance policies and consumer guides from first principles —
ingestion, hybrid retrieval, reranking, and grounded generation — with an
eval suite gating every change from day one. Nothing ships because it "looks
right"; it ships because it beats the last measured baseline.

## Architecture

```mermaid
flowchart TD
    subgraph Ingestion Pipeline
        A[Raw documents\nPDF / HTML] --> B[Parse & structure\npreserve hierarchy, defined terms]
        B --> C[Chunk\nsection/clause aware]
        C --> D[Embed]
        D --> E[(pgvector store)]
        C --> F[(Full-text / BM25 index)]
    end

    subgraph Query Pipeline
        Q[User query] --> H1[Dense retrieval\npgvector]
        Q --> H2[Sparse retrieval\nBM25]
        H1 --> R[RRF fusion]
        H2 --> R
        R --> RR[Rerank]
        RR --> G[Grounded generation\nOllama llama3.1:8b]
        G --> OUT[Answer + citations]
    end

    E -.-> H1
    F -.-> H2
```

## Phases

| Phase | Name | Status |
|-------|------|--------|
| v0 | Foundation — repo scaffold, tooling, corpus downloader | ✅ done |
| v1 | Naive RAG end to end — fixed-size chunks, dense-only retrieval, grounded generation (the intentionally naive baseline every later phase must beat) | ✅ done |
| v2 | Evaluation harness — golden dataset, retrieval + answer metrics; changes become eval-gated from here | ⬜ planned |
| v3 | Measured retrieval upgrades — structure-aware chunking, hybrid (BM25 + dense) with RRF, HNSW indexing | ⬜ planned |
| v4 | Reranking + citations | ⬜ planned |
| v5 | Generation quality — prompt iteration, groundedness checks | ⬜ planned |
| v6 | Hardening — API surface, observability | ⬜ planned |
| v7 | Final — packaging, Docker delivery | ⬜ planned |

## Design decisions

- **pgvector over a dedicated vector DB** — one fewer moving part to operate;
  revisit only if scale or query patterns demand it.
- **Native dev + Docker delivery** — develop directly against the host for
  fast iteration; ship as Docker for reproducible deployment.
- **Ollama, `llama3.1:8b`** — local-first generation, no external API
  dependency during development.
- **Eval-gated changes** — no retrieval, chunking, or prompting change lands
  without a measured effect on the golden eval set (see `eval/`).

_(This section will grow as real tradeoffs get made in later phases.)_

## Repo layout

```
src/rag_insurance/
    ingest/       # parsing, chunking
    retrieval/    # dense + sparse retrieval, fusion, reranking
    generation/   # prompting, grounded answer generation
    eval/         # eval harness code
scripts/          # one-off / operational scripts (e.g. download_data.py)
data/raw/         # downloaded source corpus (gitignored, see .gitkeep)
eval/             # golden dataset (added in v6)
NOTES/            # phase-by-phase learning notes
```

## Getting started

```bash
uv sync
cp .env.example .env
uv run python scripts/download_data.py
```

See `PROJECT_STATE.md` for current phase status and next steps.
