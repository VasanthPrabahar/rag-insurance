"""LLM-as-judge: score faithfulness and correctness of a generated answer.

Judge model is the same local llama3.1:8b via Ollama — a weak judge, which
is a documented limitation (see NOTES/phase2.md), not an endorsement.

- faithfulness: is every claim in the answer supported by the retrieved
  chunks? (1.0 = fully supported, 0.0 = not at all)
- correctness: does the answer agree with the ground truth? (1.0/0.0)

Ollama is asked for JSON output; parsing is defensive with one retry.
"""

from __future__ import annotations

import json

from pydantic import BaseModel

from rag_insurance.generation import ollama_client
from rag_insurance.retrieval.dense import RetrievedChunk

JUDGE_PROMPT = """\
You are grading an AI assistant's answer to an auto-insurance question.

Question: {question}

Ground truth answer: {ground_truth}

Context chunks the assistant was given:
{context}

The assistant's answer: {answer}

Score two things, each between 0.0 and 1.0:
1. "faithfulness": Is every factual claim in the assistant's answer supported
   by the context chunks above? 1.0 = every claim is supported, 0.5 = some
   claims unsupported, 0.0 = mostly unsupported. A refusal ("I don't know")
   is faithful (1.0). Judge ONLY against the context chunks, not your own
   knowledge.
2. "correctness": Does the assistant's answer agree with the ground truth
   answer? 1.0 = agrees on all key facts, 0.5 = partially, 0.0 = wrong or
   missing the key facts.

Respond with ONLY a JSON object: {{"faithfulness": <float>, "correctness": <float>}}
"""


class JudgeScores(BaseModel):
    faithfulness: float
    correctness: float


def _clamp(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _parse(raw: str) -> JudgeScores:
    # The model sometimes wraps JSON in prose or code fences; extract the
    # outermost braces before parsing.
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end <= start:
        raise ValueError(f"no JSON object in judge output: {raw[:200]!r}")
    data = json.loads(raw[start : end + 1])
    return JudgeScores(
        faithfulness=_clamp(data["faithfulness"]),
        correctness=_clamp(data["correctness"]),
    )


def judge_answer(
    question: str,
    ground_truth: str,
    chunks: list[RetrievedChunk],
    answer: str,
) -> JudgeScores:
    context = "\n\n".join(f"[{c.doc_name} | chunk {c.chunk_index}]\n{c.content}" for c in chunks)
    prompt = JUDGE_PROMPT.format(
        question=question, ground_truth=ground_truth, context=context, answer=answer
    )
    last_error: Exception | None = None
    for _ in range(2):  # one retry on malformed output
        raw = ollama_client.generate(prompt, json_mode=True)
        try:
            return _parse(raw)
        except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            last_error = exc
    raise RuntimeError(f"judge returned unparseable output twice: {last_error}")
