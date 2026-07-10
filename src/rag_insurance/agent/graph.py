"""LangGraph router agent.

    router -> policy_lookup  -> standard pipeline
           -> state_law      -> standard pipeline (state filter engages)
           -> multi_part     -> decompose (<=3) -> retrieve per sub-question
                                -> synthesize one cited answer over the union
           -> out_of_scope   -> refuse WITHOUT retrieval or generation
                                (saves an entire 8B generation, ~13s)

Every node appends to state["node_log"] so a run is fully reconstructable;
the CLI prints the log and structlog records it.
"""

from __future__ import annotations

import json
import time
from typing import Any, Literal, TypedDict

import structlog
from langgraph.graph import END, StateGraph

from rag_insurance.generation import ollama_client
from rag_insurance.generation.answer import (
    REFUSAL,
    VerifiedAnswer,
    grounded_answer,
)
from rag_insurance.ingest import store
from rag_insurance.retrieval.dense import RetrievedChunk
from rag_insurance.retrieval.hybrid import search

log = structlog.get_logger()

Route = Literal["policy_lookup", "state_law", "multi_part", "out_of_scope"]
ROUTES: tuple[Route, ...] = ("policy_lookup", "state_law", "multi_part", "out_of_scope")

ROUTER_PROMPT = """\
Classify this auto-insurance question into exactly one category:
- "policy_lookup": answerable from standard auto policy or consumer-guide \
content (coverages, exclusions, claims, definitions, costs)
- "state_law": asks about a specific US state's legal requirements or rules
- "multi_part": genuinely asks two or more distinct questions that need \
separate evidence
- "out_of_scope": not about auto insurance (health/boat/life insurance, \
company recommendations, anything else)

Question: {question}

Respond with ONLY JSON: {{"route": "<category>"}}"""

DECOMPOSE_PROMPT = """\
Split this multi-part auto-insurance question into at most 3 self-contained \
sub-questions, each answerable independently.

Question: {question}

Respond with ONLY JSON: {{"sub_questions": ["...", "..."]}}"""


class AgentState(TypedDict, total=False):
    question: str
    k: int
    route: Route
    sub_questions: list[str]
    chunks: list[RetrievedChunk]
    answer: VerifiedAnswer
    node_log: list[dict[str, Any]]


def _log_node(state: AgentState, node: str, t0: float, **extra: Any) -> None:
    entry = {"node": node, "ms": round((time.perf_counter() - t0) * 1000, 1), **extra}
    state.setdefault("node_log", []).append(entry)
    log.info("agent_node", **entry)


def _json_call(prompt: str) -> dict:
    raw = ollama_client.generate(prompt, json_mode=True, temperature=0.0, timeout=60.0)
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end <= start:
        raise ValueError(f"no JSON in router output: {raw[:200]!r}")
    return json.loads(raw[start : end + 1])


def classify(question: str) -> Route:
    try:
        route = _json_call(ROUTER_PROMPT.format(question=question)).get("route")
    except Exception:  # malformed output or Ollama down — fall through
        route = None
    # Unknown/malformed labels fall back to the safest route: full pipeline.
    return route if route in ROUTES else "policy_lookup"


def router_node(state: AgentState) -> AgentState:
    t0 = time.perf_counter()
    state["route"] = classify(state["question"])
    _log_node(state, "router", t0, route=state["route"])
    return state


def refuse_node(state: AgentState) -> AgentState:
    t0 = time.perf_counter()
    state["chunks"] = []
    state["answer"] = VerifiedAnswer(
        answer=REFUSAL, citations=[], refused=True, dropped_citations=[], forced_refusal=False
    )
    _log_node(state, "refuse", t0)
    return state


def standard_node(state: AgentState) -> AgentState:
    t0 = time.perf_counter()
    with store.connect() as conn:
        state["chunks"] = search(conn, state["question"], k=state.get("k", 5))
    _log_node(state, "retrieve", t0, chunks=len(state["chunks"]))
    t1 = time.perf_counter()
    state["answer"] = grounded_answer(state["question"], state["chunks"])
    _log_node(state, "generate", t1, refused=state["answer"].refused)
    return state


def decompose_node(state: AgentState) -> AgentState:
    t0 = time.perf_counter()
    try:
        subs = _json_call(DECOMPOSE_PROMPT.format(question=state["question"]))["sub_questions"]
        subs = [str(s).strip() for s in subs if str(s).strip()][:3]
    except (ValueError, KeyError, json.JSONDecodeError):
        subs = []
    state["sub_questions"] = subs or [state["question"]]
    _log_node(state, "decompose", t0, sub_questions=state["sub_questions"])
    return state


def multi_retrieve_node(state: AgentState) -> AgentState:
    t0 = time.perf_counter()
    per_sub = max(2, state.get("k", 5) // len(state["sub_questions"]))
    seen: set[tuple[str, int]] = set()
    union: list[RetrievedChunk] = []
    with store.connect() as conn:
        for sub in state["sub_questions"]:
            for chunk in search(conn, sub, k=per_sub):
                key = (chunk.doc_name, chunk.chunk_index)
                if key not in seen:
                    seen.add(key)
                    union.append(chunk)
    state["chunks"] = union
    _log_node(state, "multi_retrieve", t0, chunks=len(union))
    return state


def synthesize_node(state: AgentState) -> AgentState:
    t0 = time.perf_counter()
    state["answer"] = grounded_answer(state["question"], state["chunks"])
    _log_node(state, "synthesize", t0, refused=state["answer"].refused)
    return state


def _branch(state: AgentState) -> str:
    return state["route"]


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("router", router_node)
    graph.add_node("standard", standard_node)
    graph.add_node("decompose", decompose_node)
    graph.add_node("multi_retrieve", multi_retrieve_node)
    graph.add_node("synthesize", synthesize_node)
    graph.add_node("refuse", refuse_node)

    graph.set_entry_point("router")
    graph.add_conditional_edges(
        "router",
        _branch,
        {
            "policy_lookup": "standard",
            "state_law": "standard",  # state filter engages inside search()
            "multi_part": "decompose",
            "out_of_scope": "refuse",
        },
    )
    graph.add_edge("decompose", "multi_retrieve")
    graph.add_edge("multi_retrieve", "synthesize")
    graph.add_edge("standard", END)
    graph.add_edge("synthesize", END)
    graph.add_edge("refuse", END)
    return graph.compile()


_compiled = None


def ask(question: str, k: int = 5) -> AgentState:
    global _compiled
    if _compiled is None:
        _compiled = build_graph()
    return _compiled.invoke({"question": question, "k": k})


def retrieve_only(question: str, k: int = 5) -> AgentState:
    """The agent's retrieval path without generation — used by
    `rag eval --engine agent --skip-llm` so retrieval metrics reflect what
    the router actually feeds the generator (router/decompose LLM calls
    still run; they ARE part of the agent's retrieval)."""
    state: AgentState = {"question": question, "k": k}
    router_node(state)
    if state["route"] == "out_of_scope":
        state["chunks"] = []
    elif state["route"] == "multi_part":
        decompose_node(state)
        multi_retrieve_node(state)
    else:
        t0 = time.perf_counter()
        with store.connect() as conn:
            state["chunks"] = search(conn, state["question"], k=k)
        _log_node(state, "retrieve", t0, chunks=len(state["chunks"]))
    return state
