"""LCEL chain wrapping the existing pipeline.

The framework orchestrates; our code does the work. Each stage is one of
our own functions lifted into a RunnableLambda — no LangChain retrievers,
LLM wrappers, or prompt templates. This exists as an alternate entry point
(`rag ask --engine langchain`) for the Phase 6 comparison, not as a
replacement for the direct pipeline.
"""

from __future__ import annotations

from langchain_core.runnables import RunnableLambda, RunnablePassthrough

from rag_insurance.generation.answer import VerifiedAnswer, grounded_answer
from rag_insurance.ingest import store
from rag_insurance.retrieval.hybrid import retrieve_fused
from rag_insurance.retrieval.rewrite import rewrite_query
from rag_insurance.retrieval.state_filter import allowed_states


def _expand(state: dict) -> dict:
    queries = [state["question"]]
    rewritten = rewrite_query(state["question"])
    if rewritten:
        queries.append(rewritten)
    return {**state, "queries": queries}


def _retrieve(state: dict) -> dict:
    with store.connect() as conn:
        chunks = retrieve_fused(
            conn,
            state["question"],
            state["queries"],
            k=state.get("k", 5),
            states=allowed_states(state["question"]),
        )
    return {**state, "chunks": chunks}


def _generate(state: dict) -> dict:
    verified = grounded_answer(state["question"], state["chunks"])
    return {**state, "answer": verified}


# expand -> retrieve -> generate, exactly the direct pipeline's stages.
ask_chain = (
    RunnablePassthrough()
    | RunnableLambda(_expand)
    | RunnableLambda(_retrieve)
    | RunnableLambda(_generate)
)


def ask(question: str, k: int = 5) -> tuple[VerifiedAnswer, list]:
    result = ask_chain.invoke({"question": question, "k": k})
    return result["answer"], result["chunks"]
