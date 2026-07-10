"""Eval runner: golden set -> retrieval metrics (+ LLM judge) -> results files.

Retrieval metrics are computed for every item except out_of_scope (which has
no relevant chunks by construction). The judge scores every item; for
out_of_scope items correctness is determined programmatically: correct means
the system refused.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Literal

from pydantic import BaseModel

from rag_insurance.eval import metrics
from rag_insurance.retrieval.dense import RetrievedChunk

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
GOLDEN_SET = REPO_ROOT / "eval" / "golden_set.json"
RESULTS_DIR = REPO_ROOT / "eval" / "results"
RESULTS_MD = REPO_ROOT / "eval" / "RESULTS.md"

Category = Literal["lookup", "multi_hop", "state_specific", "out_of_scope", "adversarial"]


class GoldenItem(BaseModel):
    id: str
    question: str
    ground_truth_answer: str
    relevant_doc_names: list[str]
    relevant_hint: str | None
    category: Category
    state: str | None


class ItemResult(BaseModel):
    id: str
    category: Category
    recall_at_k: float | None
    precision_at_k: float | None
    mrr: float | None
    retrieval_ms: float | None = None
    answer_ms: float | None = None
    answer: str | None = None
    refused: bool | None = None
    citations_valid: int | None = None
    citations_dropped: int | None = None
    forced_refusal: bool | None = None
    faithfulness: float | None = None
    correctness: float | None = None
    retrieved: list[dict] = []


def load_golden_set(path: Path = GOLDEN_SET) -> list[GoldenItem]:
    data = json.loads(path.read_text())
    return [GoldenItem(**item) for item in data["items"]]


def is_refusal(answer: str) -> bool:
    return "don't know based on the provided documents" in answer.lower()


def score_retrieval(item: GoldenItem, chunks: list[RetrievedChunk], k: int) -> ItemResult:
    if item.category == "out_of_scope":
        return ItemResult(
            id=item.id, category=item.category, recall_at_k=None, precision_at_k=None, mrr=None
        )
    flags = metrics.relevance_flags(chunks, item.relevant_doc_names, item.relevant_hint)
    return ItemResult(
        id=item.id,
        category=item.category,
        recall_at_k=metrics.recall_at_k(flags, k),
        precision_at_k=metrics.precision_at_k(flags, k),
        mrr=metrics.mrr(flags),
    )


def _avg(values: list[float | None]) -> float | None:
    present = [v for v in values if v is not None]
    return round(mean(present), 4) if present else None


def aggregate(results: list[ItemResult]) -> dict:
    def block(subset: list[ItemResult]) -> dict:
        return {
            "n": len(subset),
            "recall_at_k": _avg([r.recall_at_k for r in subset]),
            "precision_at_k": _avg([r.precision_at_k for r in subset]),
            "mrr": _avg([r.mrr for r in subset]),
            "retrieval_ms": _avg([r.retrieval_ms for r in subset]),
            "answer_ms": _avg([r.answer_ms for r in subset]),
            "faithfulness": _avg([r.faithfulness for r in subset]),
            "correctness": _avg([r.correctness for r in subset]),
        }

    categories = sorted({r.category for r in results})
    per_category = {cat: block([r for r in results if r.category == cat]) for cat in categories}
    summary = {"overall": block(results), "per_category": per_category}

    judged = [r for r in results if r.citations_valid is not None]
    if judged:
        summary["citation_verification"] = {
            "valid": sum(r.citations_valid or 0 for r in judged),
            "dropped": sum(r.citations_dropped or 0 for r in judged),
            "forced_refusals": sum(1 for r in judged if r.forced_refusal),
        }
    return summary


def write_results(
    results: list[ItemResult], summary: dict, tag: str, k: int, skip_llm: bool
) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"{timestamp}_{tag}.json"
    out_path.write_text(
        json.dumps(
            {
                "tag": tag,
                "timestamp": timestamp,
                "k": k,
                "skip_llm": skip_llm,
                "summary": summary,
                "items": [r.model_dump() for r in results],
            },
            indent=2,
        )
    )
    append_results_md(summary, tag)
    return out_path


def append_results_md(summary: dict, tag: str) -> None:
    header = (
        "# Eval results\n\n"
        "One row per eval run. Retrieval metrics exclude out_of_scope items.\n\n"
        "| tag | date | recall@5 | mrr | faithfulness | correctness |\n"
        "|-----|------|----------|-----|--------------|-------------|\n"
    )
    overall = summary["overall"]

    def fmt(v: float | None) -> str:
        return f"{v:.3f}" if v is not None else "—"

    row = (
        f"| {tag} | {datetime.now().strftime('%Y-%m-%d')} | {fmt(overall['recall_at_k'])} "
        f"| {fmt(overall['mrr'])} | {fmt(overall['faithfulness'])} "
        f"| {fmt(overall['correctness'])} |\n"
    )
    if RESULTS_MD.exists():
        RESULTS_MD.write_text(RESULTS_MD.read_text() + row)
    else:
        RESULTS_MD.write_text(header + row)
