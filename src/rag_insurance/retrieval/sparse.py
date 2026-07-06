"""Sparse retrieval: Postgres full-text search over the chunk text.

websearch_to_tsquery parses the raw question (handles phrases, ignores
punctuation) and ts_rank_cd scores by cover density. This catches the
exact-term matches dense embeddings blur over — form numbers, dollar
amounts, statutory phrases.
"""

from __future__ import annotations

import psycopg

from rag_insurance.retrieval.dense import RetrievedChunk

QUERY = """
SELECT doc_name, state, doc_type, chunk_index, section_path, content,
       ts_rank_cd(content_tsv, websearch_to_tsquery('english', %(q)s)) AS score
FROM chunks
WHERE content_tsv @@ websearch_to_tsquery('english', %(q)s)
ORDER BY score DESC
LIMIT %(k)s;
"""


def retrieve(conn: psycopg.Connection, question: str, k: int = 20) -> list[RetrievedChunk]:
    rows = conn.execute(QUERY, {"q": question, "k": k}).fetchall()
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
