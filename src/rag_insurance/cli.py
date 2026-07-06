"""CLI: `rag ingest`, `rag ask`, `rag eval`."""

from __future__ import annotations

import time
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
    dense_only: bool = typer.Option(False, "--dense-only", help="Skip sparse+RRF fusion"),
    rerank: bool = typer.Option(
        False, "--rerank/--no-rerank", help="Cross-encoder reranking (default off, see NOTES)"
    ),
    rewrite: bool = typer.Option(
        True, "--rewrite/--no-rewrite", help="Policy-register query rewriting via Ollama"
    ),
) -> None:
    """Answer a question from the ingested corpus."""
    from rag_insurance.generation.answer import grounded_answer
    from rag_insurance.ingest import store
    from rag_insurance.retrieval.hybrid import search

    with store.connect() as conn:
        chunks = search(
            conn, question, k=k, dense_only=dense_only, use_rerank=rerank, use_rewrite=rewrite
        )

    verified = grounded_answer(question, chunks)
    typer.echo(verified.answer)
    if verified.citations:
        typer.echo("\nCitations:")
        for cid in verified.citations:
            chunk = chunks[cid - 1]
            typer.echo(f"  [{cid}] {chunk.doc_name} > chunk {chunk.chunk_index}")
    if verified.dropped_citations:
        typer.echo(f"\n(dropped hallucinated citation ids: {verified.dropped_citations})")
    if verified.forced_refusal:
        typer.echo("(answer had no valid citations; refusal enforced)")

    typer.echo("\n--- Retrieved chunks ---")
    for chunk in chunks:
        typer.echo(
            f"\n[{chunk.score:.4f}] {chunk.doc_name} "
            f"({chunk.state}, {chunk.doc_type}, chunk {chunk.chunk_index})"
        )
        preview = chunk.content[:300].replace("\n", " ")
        typer.echo(f"  {preview}...")


@app.command()
def eval(
    k: int = typer.Option(5, "--k", help="Chunks to retrieve per question"),
    skip_llm: bool = typer.Option(False, "--skip-llm", help="Retrieval metrics only, no judge"),
    tag: str = typer.Option("untagged", "--tag", help="Label for this run in results files"),
    dense_only: bool = typer.Option(False, "--dense-only", help="Skip sparse+RRF fusion"),
    rerank: bool = typer.Option(
        False, "--rerank/--no-rerank", help="Cross-encoder reranking (default off, see NOTES)"
    ),
    rewrite: bool = typer.Option(
        True, "--rewrite/--no-rewrite", help="Policy-register query rewriting via Ollama"
    ),
) -> None:
    """Run the golden-set evaluation."""
    from rag_insurance.eval import runner
    from rag_insurance.eval.judge import judge_answer
    from rag_insurance.generation.answer import grounded_answer
    from rag_insurance.ingest import store
    from rag_insurance.retrieval.hybrid import search

    items = runner.load_golden_set()
    results = []
    with store.connect() as conn:
        for item in items:
            t0 = time.perf_counter()
            chunks = search(
                conn,
                item.question,
                k=k,
                dense_only=dense_only,
                use_rerank=rerank,
                use_rewrite=rewrite,
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000
            result = runner.score_retrieval(item, chunks, k)
            result.retrieval_ms = round(elapsed_ms, 1)
            result.retrieved = [
                {"doc_name": c.doc_name, "chunk_index": c.chunk_index, "score": round(c.score, 4)}
                for c in chunks
            ]
            if not skip_llm:
                verified = grounded_answer(item.question, chunks)
                result.answer = verified.answer
                result.refused = verified.refused
                result.citations_valid = len(verified.citations)
                result.citations_dropped = len(verified.dropped_citations)
                result.forced_refusal = verified.forced_refusal
                if item.category == "out_of_scope" and result.refused:
                    # Refusing an unanswerable question is the correct,
                    # trivially faithful behavior; no judge needed.
                    result.faithfulness, result.correctness = 1.0, 1.0
                else:
                    scores = judge_answer(
                        item.question, item.ground_truth_answer, chunks, result.answer
                    )
                    result.faithfulness = scores.faithfulness
                    result.correctness = (
                        0.0 if item.category == "out_of_scope" else scores.correctness
                    )
            results.append(result)
            typer.echo(
                f"  {item.id} [{item.category}] "
                f"recall={result.recall_at_k} mrr={result.mrr} "
                f"faith={result.faithfulness} correct={result.correctness}"
            )

    summary = runner.aggregate(results)
    out_path = runner.write_results(results, summary, tag=tag, k=k, skip_llm=skip_llm)

    typer.echo(f"\n=== {tag} (k={k}) ===")
    typer.echo(
        f"{'category':<16}{'n':>3}{'recall@k':>10}{'prec@k':>9}{'mrr':>7}{'faith':>7}{'corr':>7}"
    )

    def fmt(v: float | None) -> str:
        return f"{v:.3f}" if v is not None else "—"

    rows = [("overall", summary["overall"])] + sorted(summary["per_category"].items())
    for name, block in rows:
        typer.echo(
            f"{name:<16}{block['n']:>3}{fmt(block['recall_at_k']):>10}"
            f"{fmt(block['precision_at_k']):>9}{fmt(block['mrr']):>7}"
            f"{fmt(block['faithfulness']):>7}{fmt(block['correctness']):>7}"
        )
    typer.echo(f"\nWrote {out_path}")


if __name__ == "__main__":
    app()
