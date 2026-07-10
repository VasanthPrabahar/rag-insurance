# Phase 6 — Framework + agentic layer notes

Nothing in the core pipeline changed. This phase wraps it two ways and
measures whether the wrapping earns its cost.

## LCEL layer: the framework orchestrates, our code works

`agent/chains.py` rebuilds ask as an LCEL chain of three RunnableLambdas —
expand → hybrid retrieve → grounded generate — each one of our existing
functions. No LangChain retrievers, LLM wrappers, or prompt templates: the
lesson of this layer is that a framework's value here is composition and
interface, not capability. Identical behavior to the direct pipeline by
construction; it exists as `--engine langchain` for comparison and as the
natural seam if the orchestration ever needs LangChain's ecosystem
(callbacks, tracing, batch APIs).

## LangGraph router: where agency actually pays

Four routes from one fast structured-output call:
- **out_of_scope → refuse without retrieval or generation.** The measured
  win: out-of-scope questions cost ~1.0s (router only) vs ~14s for a full
  pipeline pass — the agent saves an entire 8B generation by knowing when
  not to run one. Retrieval-latency evidence: oos items averaged 1021ms
  under the agent vs 1638ms retrieval-only under the pipeline (which
  retrieves and then generates a refusal anyway).
- **multi_part → decompose (≤3) → per-sub-question retrieval → one cited
  synthesis.** The unexpected result: the multi-part smoke test answered
  the TX-minimums half of its question *correctly with citations* — the
  focused sub-question "what are the minimum liability limits in Texas?"
  retrieved the cb020 evidence that the full mixed query never ranked.
  Decomposition is accidentally a fix for the g19 ranking failure:
  focused queries beat keyword-dense noise where query expansion couldn't.
- **policy_lookup / state_law → the standard pipeline** (the state filter
  already engages inside `search()`).

Router robustness is fallback-first: malformed JSON, unknown labels, or
Ollama down all route to `policy_lookup` — the safest failure is running
the full pipeline unnecessarily, never refusing a legitimate question.

Observed router behavior on the golden set: exactly g25/g26/g27 (boat,
health, best-company) short-circuited to refusal; g28 (average US premium
cost) stayed in-domain — a defensible call, since it *is* an auto-insurance
question, merely corpus-unanswerable, and the generation layer's refusal
caught it downstream. Zero in-scope questions were falsely refused — the
failure mode that would actually hurt.

## The honest comparison

Retrieval quality (full golden set, retrieval-only):

| engine | recall@5 | MRR | mean retrieval_ms |
|--------|----------|-----|-------------------|
| pipeline (v5-final) | 0.769 | 0.578 | 2856 |
| agent (v6-agent-retrieval) | 0.769 | 0.578 | 2747 |

Identical quality — 26 of 30 questions route straight into the same
pipeline, so this is expected, and the router's ~1s tax is offset by the
out-of-scope short-circuits. The agent is not a retrieval upgrade; it is a
*routing* upgrade whose wins live in latency-on-refusal and multi-part
decomposition.

Judged multi_hop comparison (6 items per engine, sequential runs):

| engine | recall@5 | MRR | faithfulness | correctness | retrieval_ms | answer_ms |
|--------|----------|-----|--------------|-------------|--------------|-----------|
| pipeline | 0.833 | 0.492 | 0.700 | 0.633 | 2188 | 15516 |
| agent | 0.833 | 0.492 | 0.817 | 0.700 | 2998 | 3102 |

**Read this table skeptically — both apparent agent wins are confounds,
and saying so is the point of the exercise:**

1. *The quality delta is judge variance, not engine quality.* The router
   classified all six multi_hop items as single questions (correctly — the
   golden set's "multi_hop" means multi-clause evidence, not multi-part
   phrasing), so both engines ran the identical pipeline and produced
   answers of identical length on 5 of 6 items. A judge scoring
   near-identical answers 0.700 vs 0.817 is the Phase 4 noise floor
   restated, now with a controlled A/A-style demonstration.
2. *The 5x answer-latency "win" is Ollama's prompt cache.* The agent run
   re-sent byte-identical generation prompts minutes after the pipeline
   run; Ollama skips re-processing a cached prompt prefix. Run order, not
   architecture. The real latency picture is the retrieval table above
   plus the out-of-scope short-circuit (~1s vs ~14s), which IS real and
   architectural.

Corollary golden-set finding: the set contains no genuinely multi-part
question, so the decompose path is exercised only by the smoke test (where
it decomposed cleanly and incidentally solved the TX-minimums retrieval
failure). A true multi-part item belongs in the next golden-set revision,
alongside the g27 relabel flagged in Phase 4.

## Langfuse: assessed, deferred

Self-hosted Langfuse v3 wants ClickHouse + Redis + MinIO + a worker — more
infrastructure than the application it would trace. Our tracing needs at
this scale are already covered: structlog emits per-request stage
latencies, and the agent logs a per-node trace into graph state (printed
by the CLI, one structlog line per node). Documented as future work for
when there are actual users to trace rather than one evaluator.
