# Phase 1 — Baseline failure analysis

Five questions run through `rag ask --k 5` against the naive pipeline
(fixed 512-token chunks, dense-only MiniLM retrieval, llama3.1:8b).
Verdicts: 2 clean passes, 1 pass with caveats, 2 failures. These become
seed cases for the Phase 2 golden dataset.

---

## Q1. What is the minimum liability coverage required in Texas? — ❌ FAIL

**Answer (abridged):** Quoted the TX ID-card form ("All drivers in Texas
must carry liability insurance...", $1,000 fine for non-compliance) and the
NAIC guide ("minimum coverage amounts are different in each state"), then
conceded the documents "do not explicitly state the minimum liability
coverage amount."

**Retrieved:** TX comparison worksheet (0.63), NAIC shopping tool (0.60),
TX ID card chunk 1 (0.60), TX ID card chunk 2 — *the Spanish-language side
of the form* (0.56), TX named-driver rule (0.56).

**Why it fails:**
- **Corpus gap:** the actual Texas 30/60/25 minimums appear nowhere in the
  corpus — no retrieval or generation change can fix a missing fact.
- **Refusal drift:** instead of the exact refusal string, the model padded
  three paragraphs of adjacent-but-nonresponsive facts. The "answer only
  from context" instruction produced *grounded rambling*, not a refusal.
- **Retrieval noise:** a Spanish-language chunk and a worksheet whose text
  is partly homeowners-insurance boilerplate both made top-5.

## Q2. Does comprehensive coverage pay if I hit a deer? — ✅ PASS

**Answer:** Yes — comprehensive pays for "impact with a bird or other
animal," quoted from the AZ premium report's coverage definitions.

**Retrieved:** AZ premium report coverage-definitions chunk (0.54) ranked
first; the rest were loosely related NAIC/AZ chunks.

**Note:** correct and grounded, and a good example of dense retrieval's
strength — "hit a deer" matched "impact with a bird or other animal" with
zero keyword overlap.

## Q3. Cracked windshield: collision or comprehensive? — ✅ PASS (caveats)

**Answer:** Glass breakage is "other than collision" (comprehensive) under
the ISO PAP, unless caused by a collision, in which case the insured may
elect to treat it as collision. Substantively correct and grounded in the
PP 00 01 09 18 Part D chunk.

**Caveats:** the model attributed the provision to "section B" (it's Part D,
paragraph B — a defined-terms/hierarchy confusion), and the top score was
only 0.49: the operative glass sentence sits mid-chunk in a 512-token window
that opens with deductible language.

## Q4. Friend drives my car and crashes — am I covered? — ❌ FAIL (retrieval)

**Answer:** "Maybe — most standard policies cover an occasional borrower,
some exclude other drivers; read your policy." Grounded in the CA consumer
guide Q&A chunk (0.64).

**Why it fails:** the corpus contains the *definitive contract answer* —
the ISO PAP Part A insuring agreement covers "any person using 'your
covered auto'" with permission — and dense retrieval never surfaced it.
The consumer guide's conversational phrasing ("Will my car be covered if
my friend is in an accident?") out-scored the policy's legal phrasing.
Naive dense retrieval systematically prefers documents that *sound like*
the question over documents that *answer* it. CA-specific guidance was also
presented as if it were general.

## Q5. What does uninsured motorist coverage protect against? — ✅ PASS

**Answer:** Damages caused by uninsured or hit-and-run drivers — medical
expenses, lost wages, pain and suffering — plus underinsured protection.
Grounded in the AZ report and ISO PAP Part C chunks.

**Note:** the top-ranked chunk (0.72) was the PAP's *exclusion* list
("'uninsured motor vehicle' does not include...") — the fixed-size chunker
split the definition from its exclusions, so the highest-scoring chunk was
the half that says what UM coverage is *not*. The model recovered because
the AZ chunk at rank 2 carried the affirmative definition.

---

## Patterns to carry into Phase 2/3

1. **Corpus gaps are invisible** until a question hits one (Q1) — the eval
   set needs answerable/unanswerable labels.
2. **Refusal behavior is soft** — "I don't know" leaks into grounded-sounding
   padding (Q1). Needs a measured groundedness/refusal metric.
3. **Register mismatch** — consumer-guide phrasing beats policy language for
   conversational questions even when the policy holds the real answer (Q4).
   This is the core case for hybrid retrieval + reranking.
4. **Chunk boundaries split definitions from exclusions** (Q5) and bury
   operative sentences mid-window (Q3) — the case for structure-aware
   chunking in Phase 3.
