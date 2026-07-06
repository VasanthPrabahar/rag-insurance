"""State detection for metadata-filtered retrieval. No LLM — regex only.

If the query names a state, retrieval is restricted to that state's
documents plus NAIC (national) material and the ISO policy specimens.
Judgment call (recorded in NOTES/phase4.md): the ISO specimens are filed
under NV/VA but the personal auto policy structure they describe is
state-agnostic, so they stay retrievable for every state's questions.
"""

from __future__ import annotations

import re

STATE_NAMES = {
    "texas": "TX",
    "california": "CA",
    "arizona": "AZ",
    "nevada": "NV",
    "virginia": "VA",
}
# Abbreviations must be standalone uppercase tokens: "TX" matches, "ca" in
# "car" or a lowercase "ca" pronoun-like token must not.
ABBREV_RE = re.compile(r"\b(TX|CA|AZ|NV|VA)\b")
NAME_RE = re.compile(r"\b(" + "|".join(STATE_NAMES) + r")\b", re.IGNORECASE)

# Always retrievable regardless of detected state.
STATE_AGNOSTIC = ["NAIC", "NV", "VA"]  # NAIC national + ISO specimens


def detect_state(question: str) -> str | None:
    name_match = NAME_RE.search(question)
    if name_match:
        return STATE_NAMES[name_match.group(1).lower()]
    abbrev_match = ABBREV_RE.search(question)
    if abbrev_match:
        return abbrev_match.group(1)
    return None


def allowed_states(question: str) -> list[str] | None:
    """States to filter retrieval to, or None for no filter."""
    state = detect_state(question)
    if state is None:
        return None
    return sorted({state, *STATE_AGNOSTIC})
