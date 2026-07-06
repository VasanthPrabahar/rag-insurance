"""Query rewriting: conversational phrasing -> policy register.

Dense and sparse retrieval both miss when the user speaks plain English but
the evidence speaks contract language ("friend drove my car" vs "permissive
use of your covered auto"). We ask a fast local LLM for formal policy TERMS
related to the question and append them to the original query — additive
expansion rather than replacement, because llama3.1:8b proved unreliable at
full rewrites (it echoed few-shot examples and drifted off-topic; see
NOTES/phase4.md). Retrieval runs with BOTH the original and the expanded
query, and RRF fusion arbitrates.

If Ollama is unreachable (e.g. CI), expansion silently degrades to the
original query so the pipeline never hard-fails on this optimization.
"""

from __future__ import annotations

import re

import httpx

from rag_insurance.generation import ollama_client

EXPAND_PROMPT = """\
List 3 to 6 formal auto-insurance policy terms that are relevant to the \
user's question, comma-separated, on one line. Use vocabulary found in \
insurance contracts and regulations. Output ONLY the terms.

Examples:
Question: My friend borrowed my car and crashed it, am I in trouble?
Terms: permissive use, your covered auto, named insured, liability coverage, insured person

Question: A deer ran into my car, who pays?
Terms: other than collision, comprehensive coverage, contact with bird or animal, deductible

Question: My car is in the shop, do I get a rental?
Terms: temporary transportation expenses, non-owned auto, temporary substitute vehicle

Question: The guy who hit me had no insurance, what now?
Terms: uninsured motorist coverage, bodily injury, uninsured motor vehicle, hit-and-run

Question: How much car insurance does the law make me buy?
Terms: minimum liability limits, financial responsibility law, property damage liability

Question: {question}
Terms:"""

_PREFIX_RE = re.compile(r"^(terms|question|q|r|a)\s*:\s*", re.IGNORECASE)


def rewrite_query(question: str) -> str | None:
    """Return the original question expanded with policy terms, or None."""
    try:
        raw = ollama_client.generate(
            EXPAND_PROMPT.format(question=question), timeout=60.0, temperature=0.0
        ).strip()
    except (httpx.HTTPError, KeyError):
        return None
    line = _PREFIX_RE.sub("", raw.splitlines()[0].strip())
    terms = [t.strip() for t in line.split(",") if 0 < len(t.strip()) <= 60]
    # Drop terms that just echo the question and cap the expansion size.
    terms = [t for t in terms if t.lower() not in question.lower()][:6]
    if not terms:
        return None
    return f"{question} {' '.join(terms)}"
