"""Idempotent delta ingestion: only new or changed files get re-processed.

A content-hash manifest table (ingest_files) records the sha256 of every
ingested file. On each run, files whose hash matches are skipped; new or
changed files are re-parsed, their old chunks deleted (chunk counts can
shrink when a document changes), and fresh chunks upserted — then the hash
row is updated in the same transaction, so a crash mid-file re-processes
that file on the next run rather than losing it.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import psycopg
from pydantic import BaseModel

FILES_SCHEMA = """
CREATE TABLE IF NOT EXISTS ingest_files (
    filename text PRIMARY KEY,
    sha256 text NOT NULL,
    ingested_at timestamptz NOT NULL DEFAULT now()
);
"""


class DeltaReport(BaseModel):
    processed: list[str]
    skipped: int
    chunks_written: int


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def init_files_schema(conn: psycopg.Connection) -> None:
    conn.execute(FILES_SCHEMA)
    conn.commit()


def detect_changes(conn: psycopg.Connection, data_dir: Path) -> tuple[list[Path], int]:
    """Return (new-or-changed files, unchanged count)."""
    recorded = dict(conn.execute("SELECT filename, sha256 FROM ingest_files").fetchall())
    changed: list[Path] = []
    unchanged = 0
    for path in sorted(data_dir.iterdir()):
        if path.name.startswith(".") or path.suffix.lower() not in (".pdf", ".html", ".htm"):
            continue
        if recorded.get(path.name) == file_sha256(path):
            unchanged += 1
        else:
            changed.append(path)
    return changed, unchanged


def ingest_deltas(conn: psycopg.Connection, files: list[Path]) -> DeltaReport:
    from rag_insurance.ingest import store
    from rag_insurance.ingest.chunker import chunk_document
    from rag_insurance.ingest.embedder import embed_texts
    from rag_insurance.ingest.parser import parse_file

    processed: list[str] = []
    chunks_written = 0
    for path in files:
        doc = parse_file(path)
        if doc is None:
            continue
        chunks = chunk_document(doc)
        # Delete-then-upsert inside one transaction: a changed document may
        # produce fewer chunks than before, and stale tails must not linger.
        with conn.transaction():
            conn.execute("DELETE FROM chunks WHERE doc_name = %s", (doc.doc_name,))
            if chunks:
                embeddings = embed_texts([chunk.content for chunk in chunks])
                store.upsert_chunks(conn, chunks, embeddings)
            conn.execute(
                """
                INSERT INTO ingest_files (filename, sha256, ingested_at)
                VALUES (%s, %s, now())
                ON CONFLICT (filename) DO UPDATE
                    SET sha256 = EXCLUDED.sha256, ingested_at = now()
                """,
                (path.name, file_sha256(path)),
            )
        processed.append(path.name)
        chunks_written += len(chunks)
    return DeltaReport(processed=processed, skipped=0, chunks_written=chunks_written)


def run_delta_ingestion(conn: psycopg.Connection, data_dir: Path) -> DeltaReport:
    from rag_insurance.ingest import store

    store.init_schema(conn)
    init_files_schema(conn)
    changed, unchanged = detect_changes(conn, data_dir)
    report = ingest_deltas(conn, changed)
    return DeltaReport(
        processed=report.processed, skipped=unchanged, chunks_written=report.chunks_written
    )
