"""pgvector storage for chunks.

Exact (sequential-scan) search only — no HNSW index yet; adding one is a
Phase 3 measured change. Upserts key on (doc_name, chunk_index) so
re-ingestion updates in place instead of duplicating.
"""

from __future__ import annotations

import os

import numpy as np
import psycopg
from dotenv import load_dotenv

from rag_insurance.ingest.chunker import Chunk
from rag_insurance.ingest.embedder import EMBEDDING_DIM

SCHEMA = f"""
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS chunks (
    id bigserial PRIMARY KEY,
    doc_name text NOT NULL,
    state text NOT NULL,
    doc_type text NOT NULL,
    chunk_index integer NOT NULL,
    section_path text NOT NULL DEFAULT '',
    content text NOT NULL,
    embedding vector({EMBEDDING_DIM}) NOT NULL,
    content_tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
    UNIQUE (doc_name, chunk_index)
);
CREATE INDEX IF NOT EXISTS chunks_content_tsv_idx ON chunks USING GIN (content_tsv);
-- HNSW (m=16, ef_construction=64 defaults): approximate NN. At this corpus
-- size the win is pedagogical — measured against exact scan in NOTES/phase3.md.
CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw_idx
    ON chunks USING hnsw (embedding vector_cosine_ops);
"""

UPSERT = """
INSERT INTO chunks (doc_name, state, doc_type, chunk_index, section_path, content, embedding)
VALUES (%s, %s, %s, %s, %s, %s, %s::vector)
ON CONFLICT (doc_name, chunk_index) DO UPDATE SET
    state = EXCLUDED.state,
    doc_type = EXCLUDED.doc_type,
    section_path = EXCLUDED.section_path,
    content = EXCLUDED.content,
    embedding = EXCLUDED.embedding;
"""


def get_postgres_url() -> str:
    load_dotenv()
    url = os.environ.get("POSTGRES_URL")
    if not url:
        raise RuntimeError("POSTGRES_URL is not set (copy .env.example to .env)")
    return url


def connect() -> psycopg.Connection:
    return psycopg.connect(get_postgres_url())


def init_schema(conn: psycopg.Connection) -> None:
    conn.execute(SCHEMA)
    conn.commit()


def to_pgvector(embedding: np.ndarray | list[float]) -> str:
    return "[" + ",".join(f"{x:.8f}" for x in embedding) + "]"


def upsert_chunks(conn: psycopg.Connection, chunks: list[Chunk], embeddings: np.ndarray) -> None:
    """Upsert chunk rows. Does NOT commit — callers own transaction scope
    (delta ingestion wraps this in an explicit transaction)."""
    if len(chunks) != len(embeddings):
        raise ValueError(f"{len(chunks)} chunks but {len(embeddings)} embeddings")
    with conn.cursor() as cur:
        cur.executemany(
            UPSERT,
            [
                (
                    chunk.doc_name,
                    chunk.state,
                    chunk.doc_type,
                    chunk.chunk_index,
                    chunk.section_path,
                    chunk.content,
                    to_pgvector(embedding),
                )
                for chunk, embedding in zip(chunks, embeddings, strict=True)
            ],
        )


def stats(conn: psycopg.Connection) -> tuple[int, int]:
    row = conn.execute("SELECT count(DISTINCT doc_name), count(*) FROM chunks").fetchone()
    assert row is not None
    return row[0], row[1]
