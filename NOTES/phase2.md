# Phase 2 — Eval harness + corpus rebalance notes

## Golden-set design rationale

Thirty questions, every one grounded in text I actually read in the corpus —
not written from imagination of what an insurance corpus "should" contain.
The category mix is deliberate: ~12 lookup (single-fact retrieval, the floor),
6 multi-hop (facts that live in different clauses, e.g. theft + the 48-hour
transportation-expense rule), 6 state-specific (tests metadata-blind dense
retrieval against TX/CA/AZ facts), 4 out-of-scope (the *correct* answer is a
refusal — a RAG system that answers everything is broken), and 2 adversarial
(questions with a false premise baked in). Five items are known-failure
probes carried forward from Phase 1: the rental-car retrieval miss, the
"friend drives my car" register mismatch, the "what is a deductible"
grounding leak, the TX-minimums corpus gap, and the split UM definition.
A relevant chunk must match on BOTH document name and a hint snippet —
document-only matching would credit retrieval for finding the right document
at the wrong clause, which is precisely the failure mode fixed-size chunking
produces. Every hint was validated (whitespace-normalized, chunk-level)
against the real chunked corpus before the set was frozen.

## Why recall upper-bounds answer quality

Generation cannot cite what retrieval never surfaced. If the evidence chunk
is absent from the context window, the model can only refuse (correct
behavior, but unhelpful) or answer from parametric memory (a grounding
violation). So recall@k is a hard ceiling on end-to-end correctness for
in-scope questions: with recall@5 = 0.5, at most half the questions can be
answered correctly *for the right reason*. This is why retrieval changes are
evaluated before generation changes in this project — improving prompts
under a low-recall retriever polishes answers to the wrong context.

## Faithfulness vs correctness — the deductible grounding leak

These two axes disagree in instructive ways, and "what is a deductible" is
the worked example. llama3.1:8b knows perfectly well what a deductible is
from pretraining. If retrieval misses the CA guide's definition chunk, the
model can still produce a *correct* answer (correctness ≈ 1.0) that is
*unfaithful* (faithfulness ≈ 0.0) — no retrieved chunk supports it. That
answer is a time bomb: the same behavior that answers "what is a deductible"
from memory will later fabricate a citation for it, or answer a question
where its memory is wrong or stale (e.g. pre-2020 state minimums). The pair
of metrics catches what either alone would miss: high correctness + low
faithfulness = grounding leak; high faithfulness + low correctness =
retrieval fed it the wrong evidence and it dutifully summarized it.

What the baseline actually showed for g04: hint-recall was 0 (the designated
CA-guide chunk wasn't retrieved), yet the answer quoted deductible language
from NAIC chunks that *were* retrieved — grounded, just in different
documents than the golden set designated. Lesson: hint-based relevance is
deliberately conservative and can under-credit retrieval; that bias is
acceptable (it never *over*-credits), but read recall dips alongside
faithfulness before declaring a retrieval regression.

## LLM-judge pitfalls

- **Weak judge model.** The judge is the same llama3.1:8b that generates
  answers — cheap and local, but it misses subtle unsupported claims and
  mis-scores borderline cases. Observed in the baseline run: on g19 (TX
  minimums, a corpus gap) the system correctly refused, and the judge scored
  that refusal correctness = 1.0 against a ground truth of "30/60/25" — a
  plain grading error; a refusal cannot match a numeric ground truth. Treat
  judge scores as a trend signal between runs, not ground truth; retrieval
  metrics are deterministic and are the primary gate.
- **Self-preference.** A model judging its own outputs scores them higher
  than an independent judge would. Same-model judging inflates absolute
  faithfulness/correctness numbers; deltas between runs are more trustworthy
  than levels.
- **Position bias.** LLM judges over-weight what appears first (in pairwise
  setups, the first candidate; in our rubric setup, the earliest context
  chunks). We mitigate by judging single answers against an explicit rubric
  and ground truth rather than pairwise ranking, and by keeping the
  out-of-scope refusal check programmatic (string match) instead of asking
  the judge.
- **JSON fragility.** Small models drift out of requested output formats;
  the judge requests Ollama's JSON mode, parses defensively (outermost-brace
  extraction), and retries once before failing loudly rather than silently
  scoring 0.

## Corpus rebalance (first measured change)

The AZ 2023 Auto Premium Report was 233 of 390 chunks — 60% of the corpus —
and almost all of it rate tables: insurer names, NAIC numbers, and premium
figures for rating scenarios. Rate-table chunks are dense-retrieval poison:
they mention every coverage term ("comprehensive", "collision", "UM/UIM",
"deductible") in close proximity, so they score well against nearly any
coverage question while containing no explanatory content. Removed it;
added the TDI Automobile Insurance Guide (cb020, HTML — fills the TX 30/60/25
corpus gap from Phase 1) and the AZ New Driver's Guide (via Wayback mirror,
difi.az.gov still Cloudflare-walled). Post-rebalance: 15 documents,
~174 chunks, largest document ≈ 18% of corpus.

One honesty note: the AZ New Driver's Guide is the 2018 edition and states
Arizona's pre-2020 minimums (15/30/10; the law moved to 25/50/15 in 2020).
The golden set deliberately avoids asking for AZ minimums so we don't
enshrine stale law as ground truth — the AZ state question asks about
uninsured-driving penalties instead, which the document states correctly.

## Results

| run | recall@5 | mrr | faithfulness | correctness |
|-----|----------|-----|--------------|-------------|
| v1-naive-dirty-corpus | 0.500 | 0.388 | 0.893 | 0.688 |
| v2-clean-corpus | 0.538 | 0.424 | 0.897 | 0.667 |
| v2-clean-corpus-labelfix (retrieval only) | 0.577 | 0.463 | — | — |

The rebalance moved retrieval, not generation: recall@5 +0.038 and MRR
+0.036 on identical golden labels, driven entirely by state_specific
(0.500 → 0.667 recall; g24's AZ-penalties evidence now exists and ranks
first). Faithfulness was flat and correctness dipped within judge noise —
expected, since generation only improves downstream of retrieval wins.

The labelfix row is a golden-set correction, not a system change: g19's
rank-1 chunk in *both* runs was the TX Comparison Worksheet, which plainly
states "30/60/25 is minimum required by law" — a relevant document I missed
when authoring the set. Since both recorded runs under-credited g19
identically, the v1→v2 delta is unaffected; future runs use the corrected
label. Meanwhile cb020's own 30/60/25 chunk ranks ~20th for the TX-minimums
query — the corpus gap is closed, but naive dense retrieval still ranks the
substantive explanation poorly. That, plus the two probes still at recall 0
(g01 rental/transportation expenses, g02 friend/permissive-use — both cases
where policy legalese loses to conversational phrasing), is the Phase 3
target list.
