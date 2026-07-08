# Phase 5 — Production service + ingestion pipeline notes

## Why models load at startup (measured)

Loading bge-base-en-v1.5 costs **6.63s**; a warm query embedding costs
**0.21s**. Loading per request would tax every request ~30x its actual
work. The FastAPI lifespan handler loads the embedder once and opens a
psycopg connection pool before the first request is accepted; the
cross-encoder (default-off) stays lazy behind its lru_cache and loads on
the first `engine.rerank=true` request. Same pattern, different trigger:
pay the load cost when the component is actually in the serving path, and
never per-request. This is also why the CLI feels slow (`rag ask` reloads
weights every invocation) — the CLI is a debugging tool; the API is the
product.

## SSE vs polling

/ask streams over Server-Sent Events rather than making clients poll a
job endpoint. With a local 8B model, generation dominates wall time —
seconds during which a polling client learns nothing and an SSE client
renders tokens as they exist. SSE over WebSockets: the stream is strictly
server→client, so HTTP + `text/event-stream` (with named events and
automatic reconnect semantics) does the job without a protocol upgrade or
a second connection lifecycle to manage.

One non-obvious piece: generation emits *JSON* (for the citation contract),
and raw JSON tokens are useless to display. `AnswerFieldExtractor` is a
small state machine that watches the token stream for the `"answer"` field
and yields only its unescaped content — real streaming, same evaluated
generation path, no second prompt format to maintain. A parse failure at
stream end degrades to a refusal-shaped final event (no retry mid-stream:
you can't un-send tokens).

## Latency budget (structlog, measured on the running service)

Measured /ask against the composed stack (API in container, Ollama native
on host, one structlog JSON line per request):

    cold first request:  expand 8770ms | retrieve 556ms | generate 13579ms | total 22906ms
    warm request:        expand 1355ms | retrieve 272ms | generate 12650ms | total 14279ms

The cold/warm gap is Ollama paging llama3.1:8b into memory plus the
container's first embedder inference. Warm, the shape is clear: generation
dominates (~89%), expansion costs ~9%, and retrieval — dense+sparse across
two query variants, RRF, over pgvector+GIN — is ~2%. The pipeline's
intelligence lives in the cheapest stage; both LLM stages are 5-50x the
cost of retrieval. Query expansion is the first thing to cut or cache if
latency matters more than the MRR it buys (Phase 6's comparison will
quantify that tradeoff).

## Idempotent delta ingestion

The `ingest_files` table records a sha256 per corpus file. Each run:
hash-compare, skip matches, and for each new/changed file — inside one
transaction — delete the document's old chunks, re-chunk, re-embed,
upsert, and update the hash row. Delete-then-insert (not just upsert)
because a changed document can produce *fewer* chunks, and upsert alone
would leave stale tails. The hash update rides the same transaction: a
crash mid-file re-processes that file next run instead of silently losing
it. Measured: first run processed 15 files / 174 chunks; second run
skipped all 15 and wrote nothing.

## Why Airflow fits (and why it's not in compose)

The ingestion job is exactly Airflow-shaped: scheduled, idempotent,
stage-structured (detect → process → summarize), with retries and run
history for free. The DAG is ~60 lines because all logic lives in
`rag_insurance.ingest.delta` — importable, testable, and runnable without
Airflow; the DAG only orchestrates.

Judgment call: Airflow stays OUT of core dependencies and docker-compose.
A scheduler + metadata DB + webserver is a heavyweight stack that would
dwarf the app it schedules, slow every `uv sync`, and complicate CI — to
run one daily job. For this corpus, `docs-only` setup (DAG header explains
`airflow standalone`) delivers the learning and the artifact without the
operational drag. At real scale (many sources, backfills, SLAs), promote
it to its own compose profile or a managed scheduler.

## Docker delivery notes

Multi-stage uv build; final image carries only the venv + source. Model
weights are NOT baked in — `HF_HOME=/data/hf` points at a named volume, so
the first startup downloads bge once and every rebuild/restart after is
fast. Ollama deliberately stays on the host (Apple Silicon GPU access from
Docker is poor); inside the container the host is `host.docker.internal`,
with `extra_hosts: host-gateway` making that work on Linux too. The image
is still torch-sized (~2GB) — "slim" here means "no weights and no build
toolchain", not small; a CPU-only torch build would halve it and is noted
as future work. Multi-arch: `docker buildx build --platform
linux/amd64,linux/arm64` (documented in the Dockerfile header).
