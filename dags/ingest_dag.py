"""Airflow DAG: delta ingestion of data/raw into pgvector.

Runs daily: hash-compare every corpus file against the ingest_files
manifest table, re-process only new/changed documents (delete old chunks,
re-chunk, re-embed, upsert), and log counts.

Airflow is deliberately NOT part of this project's core dependencies or
docker-compose (see NOTES/phase5.md). To run locally:

    uv venv .airflow-venv && source .airflow-venv/bin/activate
    pip install apache-airflow rag-insurance@file://$(pwd)
    export AIRFLOW_HOME=$(pwd)/.airflow AIRFLOW__CORE__DAGS_FOLDER=$(pwd)/dags
    airflow standalone

The task code itself is a thin wrapper over rag_insurance.ingest.delta —
the same functions the CLI and API use, so the DAG stays trivially small
and the logic stays testable without Airflow.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

from airflow.decorators import dag, task

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


@dag(
    schedule=timedelta(days=1),
    start_date=datetime(2026, 7, 1),
    catchup=False,
    default_args={"retries": 1, "retry_delay": timedelta(minutes=5)},
    tags=["rag-insurance"],
)
def rag_insurance_ingest():
    @task
    def detect() -> list[str]:
        from rag_insurance.ingest import store
        from rag_insurance.ingest.delta import detect_changes, init_files_schema

        with store.connect() as conn:
            store.init_schema(conn)
            init_files_schema(conn)
            changed, unchanged = detect_changes(conn, DATA_DIR)
        print(f"{len(changed)} new/changed, {unchanged} unchanged")
        return [str(p) for p in changed]

    @task
    def process(changed: list[str]) -> dict:
        from rag_insurance.ingest import store
        from rag_insurance.ingest.delta import ingest_deltas

        with store.connect() as conn:
            report = ingest_deltas(conn, [Path(p) for p in changed])
        return report.model_dump()

    @task
    def summarize(report: dict) -> None:
        from rag_insurance.ingest import store

        with store.connect() as conn:
            n_docs, n_chunks = store.stats(conn)
        print(
            f"processed={len(report['processed'])} "
            f"chunks_written={report['chunks_written']} | "
            f"store now: {n_docs} documents, {n_chunks} chunks"
        )

    summarize(process(detect()))


rag_insurance_ingest()
