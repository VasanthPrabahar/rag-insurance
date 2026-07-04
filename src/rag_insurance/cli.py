"""CLI: `rag ingest` and `rag ask`."""

from __future__ import annotations

from pathlib import Path

import typer

app = typer.Typer(no_args_is_help=True, add_completion=False)

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "raw"


@app.command()
def ingest(data_dir: Path = DATA_DIR) -> None:
    """Parse, chunk, embed, and store the corpus in data/raw/."""
    from rag_insurance.ingest import store
    from rag_insurance.ingest.chunker import chunk_document
    from rag_insurance.ingest.embedder import embed_texts
    from rag_insurance.ingest.parser import parse_corpus

    documents = parse_corpus(data_dir)
    typer.echo(f"Parsed {len(documents)} documents")

    with store.connect() as conn:
        store.init_schema(conn)
        for doc in documents:
            chunks = chunk_document(doc)
            if not chunks:
                typer.echo(f"  {doc.filename}: 0 chunks, skipping")
                continue
            embeddings = embed_texts([chunk.content for chunk in chunks])
            store.upsert_chunks(conn, chunks, embeddings)
            typer.echo(f"  {doc.filename}: {len(chunks)} chunks")
        n_docs, n_chunks = store.stats(conn)

    typer.echo(f"\nStore now holds {n_docs} documents, {n_chunks} chunks")


@app.command()
def ask(
    question: str,
    k: int = typer.Option(5, "--k", help="Number of chunks to retrieve"),
) -> None:
    """Answer a question from the ingested corpus."""
    from rag_insurance.generation.answer import answer
    from rag_insurance.ingest import store
    from rag_insurance.retrieval.dense import retrieve

    with store.connect() as conn:
        chunks = retrieve(conn, question, k=k)

    typer.echo(answer(question, chunks))

    typer.echo("\n--- Retrieved chunks ---")
    for chunk in chunks:
        typer.echo(
            f"\n[{chunk.score:.4f}] {chunk.doc_name} "
            f"({chunk.state}, {chunk.doc_type}, chunk {chunk.chunk_index})"
        )
        preview = chunk.content[:300].replace("\n", " ")
        typer.echo(f"  {preview}...")


if __name__ == "__main__":
    app()
