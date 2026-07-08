"""Grounded answer generation with mechanically verified citations.

The model must answer ONLY from the provided chunks, cite [chunk_id] for
every claim, and refuse when the chunks don't contain the answer — returned
as structured JSON and validated with pydantic (one retry).

Then verification happens OUTSIDE the model: cited ids must exist in the
retrieved set; hallucinated ids are dropped; an answer with claims but zero
surviving citations is replaced by the refusal. An LLM can fabricate a
plausible citation as easily as a plausible fact — the whole point is that
this check is mechanical, not model-judged.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator

from pydantic import BaseModel, ValidationError

from rag_insurance.generation import ollama_client
from rag_insurance.retrieval.dense import RetrievedChunk

REFUSAL = "I don't know based on the provided documents."

PROMPT_TEMPLATE = """\
You are an assistant answering questions about auto insurance using ONLY the
numbered context chunks below, which come from insurance policies, consumer
guides, and regulations.

Rules:
- Answer using only information stated in the context chunks.
- Answer the specific question asked, directly and in your own words — do
  not copy sentences verbatim from the chunks.
- When chunks give different rules for different situations (e.g. theft vs
  other losses), apply the rule that matches the question's situation.
- Every factual claim in your answer must cite its chunk id in square
  brackets, e.g. [2].
- If the context does not contain the answer, set "refused" to true and use
  exactly this answer text: "{refusal}"
- Do not use outside knowledge, even if you are confident.

Respond with ONLY a JSON object:
{{"answer": "<answer text with [n] citations>",
 "citations": [<chunk ids you cited>], "refused": <true|false>}}

Context chunks:
{context}

Question: {question}

JSON:"""


class GroundedAnswer(BaseModel):
    answer: str
    citations: list[int]
    refused: bool


class VerifiedAnswer(BaseModel):
    answer: str
    citations: list[int]  # ids that survived verification
    refused: bool
    dropped_citations: list[int]  # hallucinated ids removed mechanically
    forced_refusal: bool  # answer had claims but zero valid citations


def build_prompt(question: str, chunks: list[RetrievedChunk]) -> str:
    context = "\n\n".join(
        f"[{i}] ({chunk.doc_name} | {chunk.state} | {chunk.doc_type} | chunk {chunk.chunk_index})\n"
        f"{chunk.content}"
        for i, chunk in enumerate(chunks, start=1)
    )
    return PROMPT_TEMPLATE.format(refusal=REFUSAL, context=context, question=question)


def _parse(raw: str) -> GroundedAnswer:
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end <= start:
        raise ValueError(f"no JSON object in output: {raw[:200]!r}")
    data = json.loads(raw[start : end + 1])
    citations = [int(c) for c in data.get("citations", []) if isinstance(c, (int, str))]
    return GroundedAnswer(
        answer=str(data.get("answer", "")),
        citations=citations,
        refused=bool(data.get("refused", False)),
    )


def _generate(prompt: str) -> GroundedAnswer:
    last_error: Exception | None = None
    for _ in range(2):  # one retry on malformed output
        raw = ollama_client.generate(prompt, json_mode=True, temperature=0.0)
        try:
            return _parse(raw)
        except (ValueError, KeyError, TypeError, json.JSONDecodeError, ValidationError) as exc:
            last_error = exc
    raise RuntimeError(f"generation returned unparseable JSON twice: {last_error}")


def verify(generated: GroundedAnswer, n_chunks: int) -> VerifiedAnswer:
    """Mechanical citation verification — no model involved."""
    valid = [c for c in generated.citations if 1 <= c <= n_chunks]
    dropped = [c for c in generated.citations if c not in valid]

    if generated.refused:
        return VerifiedAnswer(
            answer=REFUSAL,
            citations=[],
            refused=True,
            dropped_citations=dropped,
            forced_refusal=False,
        )
    if not valid:
        # Claims with zero surviving citations are ungroundable — refuse.
        return VerifiedAnswer(
            answer=REFUSAL,
            citations=[],
            refused=True,
            dropped_citations=dropped,
            forced_refusal=True,
        )
    return VerifiedAnswer(
        answer=generated.answer,
        citations=valid,
        refused=False,
        dropped_citations=dropped,
        forced_refusal=False,
    )


def grounded_answer(question: str, chunks: list[RetrievedChunk]) -> VerifiedAnswer:
    generated = _generate(build_prompt(question, chunks))
    return verify(generated, n_chunks=len(chunks))


class AnswerFieldExtractor:
    """Incrementally extract the "answer" string from a streaming JSON object.

    The generator emits JSON ({"answer": "...", ...}); streaming raw JSON to a
    client is useless for display, so this state machine watches the byte
    stream for the answer field and yields its unescaped content as it
    arrives — real token streaming with the same evaluated generation path.
    """

    _OPEN = re.compile(r'"answer"\s*:\s*"')

    def __init__(self) -> None:
        self._buffer = ""
        self._in_answer = False
        self._escape = False
        self._done = False

    def feed(self, delta: str) -> str:
        if self._done:
            return ""
        out: list[str] = []
        self._buffer += delta
        if not self._in_answer:
            match = self._OPEN.search(self._buffer)
            if not match:
                return ""
            self._in_answer = True
            self._buffer = self._buffer[match.end() :]
        text = self._buffer
        self._buffer = ""
        for ch in text:
            if self._escape:
                out.append({"n": "\n", "t": "\t", '"': '"', "\\": "\\"}.get(ch, ch))
                self._escape = False
            elif ch == "\\":
                self._escape = True
            elif ch == '"':
                self._done = True
                break
            else:
                out.append(ch)
        return "".join(out)


def stream_grounded_answer(
    question: str, chunks: list[RetrievedChunk]
) -> Iterator[tuple[str, str | VerifiedAnswer]]:
    """Yield ("token", text) deltas, then one ("final", VerifiedAnswer).

    No retry here (unlike grounded_answer): a stream can't be transparently
    restarted after tokens were already sent. A parse failure at the end
    yields a refusal-shaped final event instead.
    """
    prompt = build_prompt(question, chunks)
    extractor = AnswerFieldExtractor()
    raw_parts: list[str] = []
    for delta in ollama_client.generate_stream(prompt, json_mode=True, temperature=0.0):
        raw_parts.append(delta)
        text = extractor.feed(delta)
        if text:
            yield ("token", text)
    try:
        generated = _parse("".join(raw_parts))
        yield ("final", verify(generated, n_chunks=len(chunks)))
    except (ValueError, KeyError, TypeError, json.JSONDecodeError, ValidationError):
        yield (
            "final",
            VerifiedAnswer(
                answer=REFUSAL,
                citations=[],
                refused=True,
                dropped_citations=[],
                forced_refusal=True,
            ),
        )
