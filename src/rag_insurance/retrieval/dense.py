"""Dense retrieval: embed the query, top-k by cosine similarity in pgvector."""

from __future__ import annotations

import psycopg
from pydantic import BaseModel

from rag_insurance.ingest.embedder import embed_query
from rag_insurance.ingest.store import to_pgvector

QUERY = """
SELECT doc_name, state, doc_type, chunk_index, section_path, content,
       1 - (embedding <=> %(q)s::vector) AS score
FROM chunks
WHERE %(states)s::text[] IS NULL OR state = ANY(%(states)s)
ORDER BY embedding <=> %(q)s::vector
LIMIT %(k)s;
"""


class RetrievedChunk(BaseModel):
    doc_name: str
    state: str
    doc_type: str
    chunk_index: int
    section_path: str = ""
    content: str
    score: float


def retrieve(
    conn: psycopg.Connection,
    question: str,
    k: int = 5,
    states: list[str] | None = None,
) -> list[RetrievedChunk]:
    query_vec = to_pgvector(embed_query(question))
    rows = conn.execute(QUERY, {"q": query_vec, "k": k, "states": states}).fetchall()
    return [
        RetrievedChunk(
            doc_name=row[0],
            state=row[1],
            doc_type=row[2],
            chunk_index=row[3],
            section_path=row[4],
            content=row[5],
            score=row[6],
        )
        for row in rows
    ]
