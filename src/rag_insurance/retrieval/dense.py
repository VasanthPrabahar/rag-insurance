"""Dense retrieval: embed the query, top-k by cosine similarity in pgvector."""

from __future__ import annotations

import psycopg
from pydantic import BaseModel

from rag_insurance.ingest.embedder import embed_query
from rag_insurance.ingest.store import to_pgvector

QUERY = """
SELECT doc_name, state, doc_type, chunk_index, content,
       1 - (embedding <=> %(q)s::vector) AS score
FROM chunks
ORDER BY embedding <=> %(q)s::vector
LIMIT %(k)s;
"""


class RetrievedChunk(BaseModel):
    doc_name: str
    state: str
    doc_type: str
    chunk_index: int
    content: str
    score: float


def retrieve(conn: psycopg.Connection, question: str, k: int = 5) -> list[RetrievedChunk]:
    query_vec = to_pgvector(embed_query(question))
    rows = conn.execute(QUERY, {"q": query_vec, "k": k}).fetchall()
    return [
        RetrievedChunk(
            doc_name=row[0],
            state=row[1],
            doc_type=row[2],
            chunk_index=row[3],
            content=row[4],
            score=row[5],
        )
        for row in rows
    ]
