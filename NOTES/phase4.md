# Phase 4 — Query intelligence + grounded generation notes

## Register mismatch and why rewriting (partially) fixes it

The corpus speaks two languages: consumer guides ("will my car be covered
if my friend is in an accident?") and contract legalese ("any person using
'your covered auto'"). Users ask in the first; the binding answers live in
the second. Both retrievers fail on the gap for different reasons — dense
embeddings put conversational queries nearer conversational text, and BM25
literally cannot match words that aren't there ("friend" appears nowhere in
Part A). Query rewriting bridges the gap *before* retrieval by translating
the query into the corpus's vocabulary.

Implementation note learned the hard way: llama3.1:8b could not reliably do
full query *rewrites* — at temperature 0 it echoed few-shot examples and
drifted off-topic (a UM question rewritten into permissive-use language).
Switched to additive *term expansion*: the model only lists policy terms,
which get appended to the original query. A bad expansion adds noise the
original query anchors against; a bad rewrite replaces the query entirely.
Small models get the degraded-gracefully design, not the elegant one.

Measured: MRR 0.615 → 0.663, recall flat, zero items regressed. Latency
93 → 1547ms/query (one 8B call). KEPT.

**The honest miss:** g02 (friend/permissive use) did NOT flip. Expansion
moved its evidence chunk from dense rank 28 to rank 14 — real progress,
not enough. The chunk is a fixed 512-token wall of Part A text whose
embedding is dominated by exclusion clauses, and its key phrases ("your
covered auto", "named insured") appear in nearly every policy chunk, so
BM25 gets no discrimination either. Even reranking on top of rewriting
failed to lift it (0.731/0.506 overall — the Phase 3 truncation pathology
again). g02 is a *chunk representation* problem. Ironically this is the one
case structure-aware chunking (Phase 3, reverted) would likely have fixed —
a focused "PART A > Insuring Agreement" chunk would rank far better. The
right revisit is targeted: structure-chunk only the insuring agreements,
not the whole corpus.

## HyDE as the generalization

Term expansion is a hand-rolled special case of a general idea: don't embed
the *question*, embed something that looks like the *answer*. HyDE
(Hypothetical Document Embeddings) asks the LLM to write a fake policy
passage answering the question, then retrieves with the fake passage's
embedding — closing the register gap from the same side we do, but without
maintaining a term vocabulary in the prompt. We chose explicit terms over
HyDE because the failure mode is inspectable (you can read the terms) and
the 8B model is more reliable listing vocabulary than authoring convincing
fake contract text. At a larger model size, HyDE is the natural upgrade.

## State filtering (kept, flat)

Regex-only state detection gates retrieval to {named state, NAIC, ISO
specimens}. Judgment call recorded: the ISO specimens are filed under NV/VA
but describe the state-agnostic personal auto policy structure, so they
stay retrievable for every state's questions. Metrics were flat — dense +
BM25 were already surfacing right-state documents for this golden set —
but the filter is free (no LLM, no measurable latency) and structurally
prevents cross-state contamination, a failure class the golden set
under-probes (it never asks a TX question whose distractor is a
near-identical CA chunk). Kept on that argument; would revert if it ever
cost recall.

## Why mechanical citation verification is necessary

The worked example: in Phase 2, "what is a deductible" produced a correct
answer with recall=0 — answered from pretraining, and this class of
behavior later fabricates citations. The judge scored it faithful anyway
(0.9): an LLM judge can be fooled by fluent, *plausible* grounding claims,
because a model that can fabricate a fact can fabricate its provenance in
the same breath.

The fix is a check no model touches: the generator must emit structured
JSON citing numbered chunk ids; code (not a model) verifies every cited id
exists in the retrieved set, drops hallucinated ids, and replaces any
answer that has claims but zero surviving citations with the refusal. The
LLM can still lie about *what a chunk says* — that's the judge's job to
catch — but it can no longer invent *sources*, and an answer with no
sources never ships. Phase 4 spot-check: the deductible probe now returns
a grounded answer citing a real NAIC chunk that survives verification.

## Citation-forcing has a side effect: extractive quoting

The first judged run with the citation prompt (v4-citations) dropped
faithfulness 0.903 → 0.828, concentrated in multi_hop (0.867 → 0.642).
Reading the answers showed why: forced to cite every claim, the 8B model
drifted into *copying* policy sentences instead of answering — g17 pasted
the limit-of-liability clause verbatim, g13 cited the right chunk but
applied the 24-hour non-theft rule to a theft question, g14 answered a
different question than asked. Citation pressure turned a synthesizer into
a quoter.

One measured prompt iteration fixed it: explicit instructions to answer in
your own words, answer the specific question, and apply the rule matching
the question's situation when a chunk contains several. Multi_hop-only
recheck: faithfulness 0.642 → 0.800, correctness 0.308 → 0.733 (g13 now
picks the 48-hour theft rule). Lesson: grounding constraints and answer
quality trade off through the prompt, and only a judged eval sees it —
retrieval metrics were identical across both prompts.

Also observed: rewrite-based retrieval adds run-to-run nondeterminism
(v4-citations MRR 0.578 vs v4-state-filter 0.663 on identical config) —
Ollama at temperature 0 is not bit-stable across runs, and different
expansions shuffle fused rankings. Judge deltas under ~0.05 and MRR deltas
under ~0.08 should be read against this noise floor.

## Refusal as a feature

An honest "I don't know based on the provided documents" is a correct
output, not a failure — it's what makes the system trustworthy on the 4
out-of-scope golden items and on corpus gaps. The refusal path now has
three triggers: the model refuses on its own; the model answers but all
its citations are fake (forced refusal); or retrieval returns nothing
usable. Forcing refusal on citation failure converts silent hallucination
into visible abstention — the eval sees `forced_refusal: true` and the
per-run verification counts (valid/dropped/forced) quantify how often the
generator tried to overreach.

## Final numbers and two honest footnotes

v4-final: recall@5 0.769, MRR 0.578, faithfulness 0.860, correctness 0.727
(vs v3-final 0.769 / 0.615 / 0.903 / 0.673). Correctness rose 0.054 —
multi_hop went 0.483 → 0.833 after the prompt iteration — and every one of
the 29 emitted citations survived mechanical verification across both
judged runs (zero fabricated ids, zero forced refusals).

Footnote 1: faithfulness landed 0.043 below the 0.903 target. Two judged
runs of the same config differed by more than this (see the noise-floor
note above), and the citation-style answers are structurally different
prose than what the judge scored in v3 — we report the number as measured
and note that the *mechanical* grounding guarantee (verified citations)
is stronger evidence than the judge delta is against.

Footnote 2: g27 ("which company is best?") no longer refuses — it answers
"there is no single best company, shop around and compare," which is
literally what the NAIC guides say. Our out_of_scope label scores any
non-refusal as 0. The label encodes "refuse subjective questions"; the
system now does something arguably better — grounded meta-advice. Kept the
label and the score hit; flagged for the next golden-set revision.

## The keyword-density precision problem

The TX named-driver statute commentary (31 chunks of regulatory prose
saturated with "Texas", "liability", "coverage", "policy") outranks the
actually-responsive cb020 chunk for the TX-minimums question — it wins
both retrievers by keyword saturation while containing no answer. Query
expansion did not shift it (cb020's 30/60/25 chunk still misses top-5).
What would fix it, in rough order of cost: (1) doc-type priors — boost
guides over statute *commentary* for consumer questions; (2) a
longer-context reranker that actually reads the candidates
(bge-reranker-base; the Phase 3 truncation issue is model-specific, not
inherent); (3) targeted structure-chunking of the statute so its 31 chunks
stop flooding the candidate pool. Left as future work with the failure
precisely pinned.

## The 8B generation ceiling — tested, not present (here)

Spot-check: with the cb020 evidence chunk placed directly in context, the
citation-forcing prompt extracts "at least $30,000 ... $60,000 ... $25,000"
with a valid citation, exactly right. g19's end-to-end failure is entirely
retrieval ranking, not generation. No model swap needed for this case; the
config stays one line if that changes.
