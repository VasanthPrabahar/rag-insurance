# Phase 0 — Foundation notes

- **Corpus selection rationale**: documents were chosen to span consumer-facing
  guides, a policy-adjacent form (ID card), and a state regulation, across
  three states (TX, CA, AZ) plus NAIC national material. Mixing doc types on
  purpose — consumer guides use plain language while statutes/forms use dense
  legal/regulatory language — so early ingestion and retrieval work is forced
  to handle both registers, not just the easy one.

- **Why policies are hard for naive RAG**: auto policies are not flat prose.
  A clause's meaning depends on (a) hierarchy — an exclusion in Part D can be
  overridden by an endorsement filed separately, (b) defined terms — words
  like "you," "insured," or "occurrence" are capitalized and mean exactly
  what a definitions section says, not their plain-English sense, and
  (c) cross-references — "subject to the Limit of Liability shown in the
  Declarations" or "as described in Part A" require resolving a pointer
  elsewhere in the document. Fixed-size chunking destroys all three: it
  splits definitions from their use, breaks clauses mid-hierarchy, and loses
  the anchor a cross-reference points to.

- **ISO personal auto policy structure (Parts A-D)**: the standard ISO PAP
  is organized as Part A (Liability Coverage), Part B (Medical Payments
  Coverage), Part C (Uninsured/Underinsured Motorists Coverage), and Part D
  (Coverage for Damage to Your Auto), followed by general Duties After an
  Accident/Loss and General Provisions sections that apply across all parts.
  Each Part has its own insuring agreement, exclusions, and limits — so a
  question like "am I covered for this?" often needs the right Part
  identified before the right clause within it.

- **Repo conventions**: src-layout uv project (`rag_insurance` package),
  ruff for lint/format at 100-char lines, pre-commit enforces both before
  commit, `PROJECT_STATE.md` is the living status doc updated every phase,
  and no RAG logic exists yet — this phase is scaffold and corpus only.

- **Source availability**: TDI, CDI, and NAIC document URLs were all directly
  reachable and served correctly typed PDFs. Arizona DIFI (`difi.az.gov`) sits
  behind a domain-wide Cloudflare bot challenge that blocks plain HTTP clients
  (curl/httpx) on every path tried, including the site root — not just the
  PDF links. Worked around by sourcing the same two AZ documents from their
  Wayback Machine mirrors, which serve identical content over plain HTTPS.
  If direct `difi.az.gov` access is needed later, it will require a headless
  browser or a Cloudflare-solving proxy.
