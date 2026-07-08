"""Store/retrieve integration test. Skips if the database is unreachable."""

import numpy as np
import pytest

from rag_insurance.ingest import store
from rag_insurance.ingest.chunker import Chunk
from rag_insurance.ingest.embedder import EMBEDDING_DIM

TEST_DOC = "__test_roundtrip__"


@pytest.fixture
def conn():
    try:
        connection = store.connect()
    except Exception as exc:
        pytest.skip(f"database unreachable: {exc}")
    store.init_schema(connection)
    yield connection
    connection.execute("DELETE FROM chunks WHERE doc_name = %s", (TEST_DOC,))
    connection.commit()
    connection.close()


def make_chunks(n: int) -> tuple[list[Chunk], np.ndarray]:
    rng = np.random.default_rng(42)
    embeddings = rng.normal(size=(n, EMBEDDING_DIM))
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    chunks = [
        Chunk(
            doc_name=TEST_DOC,
            state="TX",
            doc_type="guide",
            chunk_index=i,
            content=f"test chunk {i}",
        )
        for i in range(n)
    ]
    return chunks, embeddings


def count_test_chunks(conn) -> int:
    row = conn.execute("SELECT count(*) FROM chunks WHERE doc_name = %s", (TEST_DOC,)).fetchone()
    return row[0]


def test_upsert_and_nearest_neighbor(conn):
    chunks, embeddings = make_chunks(3)
    store.upsert_chunks(conn, chunks, embeddings)
    conn.commit()

    # Nearest neighbor to embedding 1 (restricted to test rows) is chunk 1.
    row = conn.execute(
        """
        SELECT chunk_index FROM chunks
        WHERE doc_name = %s
        ORDER BY embedding <=> %s::vector
        LIMIT 1
        """,
        (TEST_DOC, store.to_pgvector(embeddings[1])),
    ).fetchone()
    assert row[0] == 1


def test_reingestion_does_not_duplicate(conn):
    chunks, embeddings = make_chunks(3)
    store.upsert_chunks(conn, chunks, embeddings)
    assert count_test_chunks(conn) == 3
    store.upsert_chunks(conn, chunks, embeddings)
    assert count_test_chunks(conn) == 3
