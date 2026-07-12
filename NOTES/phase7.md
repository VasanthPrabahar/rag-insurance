# Phase 7 — Golden-set revision + final polish notes

## The revision

From v4/v6 findings: g27 ("which company is best?") relabeled from
out_of_scope to in-domain lookup — the corpus grounds a real answer
("no single best; shop around, compare quotes, check complaint records"),
and refusing it was the weaker behavior. Two genuinely multi-part
questions added (g31 AZ-penalties + hail; g32 CA good-driver + bail
bonds), each with evidence spanning multiple documents and per-part
hints. Metrics extended: multi-part recall@k is *hint coverage* (finding
one of two parts = 0.5), because hit@k would over-credit partial evidence.

## Revised-set results (retrieval-only, 32 items)

| engine | recall@5 | MRR | mean retrieval_ms |
|--------|----------|-----|-------------------|
| pipeline (v7-pipeline-revised) | 0.776 | 0.594 | 1687 |
| agent (v7-agent-revised) | 0.741 | 0.570 | 2314 |

**The pipeline now beats the agent, and the reason is the best finding of
the phase:** the router still classifies g27 as out_of_scope and refuses
in ~0.6s without retrieving — recall 0 on an item the revised set says is
answerable. The golden set changed its philosophy; the router's prompt
still encodes the old one. Agent routing *bakes evaluation judgments into
prompts*, creating a coupling that the direct pipeline simply doesn't
have. (The fix is a one-line router-prompt edit — deliberately NOT made
in this phase, because the point of the record is that the disagreement
happened and where it lives.)

**Decompose, exercised for real, did not help.** g31: both engines hit
full coverage — the single fused query already surfaces both documents,
so the agent's 5.0s decompose+multi-retrieve bought nothing over the
pipeline's 1.5s. g32: both engines score 0.5 — the bail-bonds evidence
ranks, the CA good-driver chunk misses top-5 under both strategies, and
decomposition costs 8.2s of retrieval to reach the same miss. On this
corpus, with hybrid+expansion already strong, decomposition is latency
without recall. Where it DID show value was the Phase 6 smoke test
(TX-minimums extraction) — i.e., its value is situational and currently
unpredictable, which is exactly what should be written down rather than
smoothed over.

## Demo

`demo/app.py` (Streamlit, ~100 lines): chat against the API's SSE stream,
tokens rendered live, citations as expanders showing the actual chunk
text (the API's retrieved metadata now carries chunk content), state
dropdown mapping to `filters.state`, engine toggle pipeline/agent.
`demo/DEMO_SCRIPT.md` holds the 3-beat GIF storyboard.

## Closing status

v1.0. Carried-forward future work, in priority order:
1. Targeted structure-chunking of policy insuring agreements (the g02 fix)
2. Longer-context reranker (bge-reranker-base) to retire the Phase 3
   truncation pathology
3. Larger judge model to shrink the ~0.05 judge noise floor
4. Router prompt/golden-set alignment (the g27 coupling above)
5. Doc-type priors for the keyword-density precision problem
