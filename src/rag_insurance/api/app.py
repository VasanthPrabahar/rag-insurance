"""FastAPI service for the auto-insurance RAG.

Models load ONCE at startup via the lifespan handler (the bge embedder
alone costs ~6.6s to load vs ~0.2s per warm query — a 30x per-request tax
if loaded lazily), alongside a psycopg connection pool. /ask streams the
answer over SSE and finishes with a JSON event carrying citations and a
per-stage latency breakdown. One structlog JSON line per request.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from contextlib import asynccontextmanager
from pathlib import Path

import structlog
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from psycopg_pool import ConnectionPool

from rag_insurance.api.schemas import (
    AskFinal,
    AskRequest,
    Citation,
    HealthResponse,
    IngestResponse,
    LatencyBreakdown,
    RetrievedChunkMeta,
    StatsResponse,
)
from rag_insurance.generation import ollama_client
from rag_insurance.generation.answer import VerifiedAnswer, stream_grounded_answer
from rag_insurance.ingest import store
from rag_insurance.retrieval.hybrid import retrieve_fused
from rag_insurance.retrieval.rewrite import rewrite_query
from rag_insurance.retrieval.state_filter import STATE_AGNOSTIC, allowed_states

structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.processors.JSONRenderer(),
    ]
)
log = structlog.get_logger()

DATA_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data" / "raw"


@asynccontextmanager
async def lifespan(app: FastAPI):
    from rag_insurance.ingest.embedder import get_model

    t0 = time.perf_counter()
    get_model()  # load embedding weights once
    app.state.pool = ConnectionPool(store.get_postgres_url(), min_size=1, max_size=8)
    app.state.pool.wait()
    log.info("startup", model_load_s=round(time.perf_counter() - t0, 2))
    yield
    app.state.pool.close()


app = FastAPI(title="rag-insurance", lifespan=lifespan)


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _ask_alternate_engine(request: AskRequest) -> Iterator[str]:
    """langchain/agent modes: synchronous run, single final SSE event."""
    t_start = time.perf_counter()
    if request.engine.mode == "agent":
        from rag_insurance.agent import graph

        state = graph.ask(request.question, k=request.k)
        verified, chunks = state["answer"], state.get("chunks", [])
    else:
        from rag_insurance.agent import chains

        verified, chunks = chains.ask(request.question, k=request.k)
    total_ms = round((time.perf_counter() - t_start) * 1000, 1)
    final = AskFinal(
        answer=verified.answer,
        refused=verified.refused,
        forced_refusal=verified.forced_refusal,
        citations=[
            Citation(
                chunk_id=cid,
                doc_name=chunks[cid - 1].doc_name,
                chunk_index=chunks[cid - 1].chunk_index,
                section_path=chunks[cid - 1].section_path,
            )
            for cid in verified.citations
        ],
        retrieved=[
            RetrievedChunkMeta(
                chunk_id=i, doc_name=c.doc_name, chunk_index=c.chunk_index, score=c.score
            )
            for i, c in enumerate(chunks, start=1)
        ],
        latency=LatencyBreakdown(
            expand_ms=0.0, retrieve_ms=0.0, generate_ms=0.0, total_ms=total_ms
        ),
    )
    log.info(
        "ask",
        engine=request.engine.mode,
        question=request.question[:120],
        refused=verified.refused,
        total_ms=total_ms,
    )
    yield _sse("final", final.model_dump())


@app.post("/ask")
def ask(request: AskRequest) -> StreamingResponse:
    if request.engine.mode != "pipeline":
        return StreamingResponse(_ask_alternate_engine(request), media_type="text/event-stream")

    def event_stream() -> Iterator[str]:
        t_start = time.perf_counter()

        # Stage 1: expand
        queries = [request.question]
        if request.engine.rewrite and not request.engine.dense_only:
            rewritten = rewrite_query(request.question)
            if rewritten:
                queries.append(rewritten)
        expand_ms = (time.perf_counter() - t_start) * 1000

        # Stage 2: retrieve
        if request.filters.state:
            states = sorted({request.filters.state.upper(), *STATE_AGNOSTIC})
        else:
            states = allowed_states(request.question)
        t_retrieve = time.perf_counter()
        with app.state.pool.connection() as conn:
            if request.engine.dense_only:
                from rag_insurance.retrieval import dense

                chunks = dense.retrieve(conn, request.question, k=request.k, states=states)
            else:
                chunks = retrieve_fused(
                    conn,
                    request.question,
                    queries,
                    k=request.k,
                    states=states,
                    use_rerank=request.engine.rerank,
                )
        retrieve_ms = (time.perf_counter() - t_retrieve) * 1000

        # Stage 3: generate (streamed)
        t_generate = time.perf_counter()
        verified: VerifiedAnswer | None = None
        for kind, payload in stream_grounded_answer(request.question, chunks):
            if kind == "token":
                yield _sse("token", {"text": payload})
            else:
                verified = payload  # type: ignore[assignment]
        generate_ms = (time.perf_counter() - t_generate) * 1000
        assert verified is not None

        final = AskFinal(
            answer=verified.answer,
            refused=verified.refused,
            forced_refusal=verified.forced_refusal,
            citations=[
                Citation(
                    chunk_id=cid,
                    doc_name=chunks[cid - 1].doc_name,
                    chunk_index=chunks[cid - 1].chunk_index,
                    section_path=chunks[cid - 1].section_path,
                )
                for cid in verified.citations
            ],
            retrieved=[
                RetrievedChunkMeta(
                    chunk_id=i, doc_name=c.doc_name, chunk_index=c.chunk_index, score=c.score
                )
                for i, c in enumerate(chunks, start=1)
            ],
            latency=LatencyBreakdown(
                expand_ms=round(expand_ms, 1),
                retrieve_ms=round(retrieve_ms, 1),
                generate_ms=round(generate_ms, 1),
                total_ms=round((time.perf_counter() - t_start) * 1000, 1),
            ),
        )
        log.info(
            "ask",
            question=request.question[:120],
            k=request.k,
            refused=verified.refused,
            citations=len(verified.citations),
            expand_ms=final.latency.expand_ms,
            retrieve_ms=final.latency.retrieve_ms,
            generate_ms=final.latency.generate_ms,
            total_ms=final.latency.total_ms,
        )
        yield _sse("final", final.model_dump())

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/ingest")
def ingest() -> IngestResponse:
    from rag_insurance.ingest.chunker import chunk_document
    from rag_insurance.ingest.embedder import embed_texts
    from rag_insurance.ingest.parser import parse_corpus

    t0 = time.perf_counter()
    documents = parse_corpus(DATA_DIR)
    with app.state.pool.connection() as conn:
        store.init_schema(conn)
        for doc in documents:
            chunks = chunk_document(doc)
            if chunks:
                embeddings = embed_texts([chunk.content for chunk in chunks])
                store.upsert_chunks(conn, chunks, embeddings)
                conn.commit()
        n_docs, n_chunks = store.stats(conn)
    log.info(
        "ingest", documents=n_docs, chunks=n_chunks, elapsed_s=round(time.perf_counter() - t0, 1)
    )
    return IngestResponse(documents=n_docs, chunks=n_chunks)


@app.get("/health")
def health() -> HealthResponse:
    db_ok = False
    try:
        with app.state.pool.connection() as conn:
            conn.execute("SELECT 1")
        db_ok = True
    except Exception:
        pass
    ollama_ok = ollama_client.is_reachable()
    return HealthResponse(
        status="ok" if (db_ok and ollama_ok) else "degraded", db=db_ok, ollama=ollama_ok
    )


@app.get("/stats")
def stats() -> StatsResponse:
    with app.state.pool.connection() as conn:
        n_docs, n_chunks = store.stats(conn)
    return StatsResponse(documents=n_docs, chunks=n_chunks)
