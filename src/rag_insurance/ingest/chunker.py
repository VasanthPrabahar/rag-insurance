"""Structure-aware chunking (Phase 3).

ISO policy specimens are split on their real hierarchy — PART A-F headings,
sections within each part (INSURING AGREEMENT, EXCLUSIONS, LIMIT OF
LIABILITY, ...), and individual definitions — so an exclusion chunk stays
attached to its parent coverage part via section_path, and each defined
term is retrievable on its own. Other documents split on detected headings
where present and otherwise paragraph-pack to the target size. Oversized
sections fall back to token windows that inherit the section path.

The section path is also prepended to the chunk content so both the
embedder and full-text search see the hierarchy context.
"""

from __future__ import annotations

import re
from functools import lru_cache

from pydantic import BaseModel
from transformers import AutoTokenizer

from rag_insurance.ingest.parser import ParsedDocument
from rag_insurance.manifest import DocType

TOKENIZER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_TOKENS = 512
OVERLAP_TOKENS = 64

PART_RE = re.compile(r"^PART [A-F]\b")
DEFINITION_ITEM_RE = re.compile(r"^[A-Z]\.\s")
NOISE_RE = re.compile(r"^(PP \d|Page \d|\d+$|Copyright)")
QUOTED_TERM_RE = re.compile(r'"([^"]+)"')


class Chunk(BaseModel):
    doc_name: str
    state: str
    doc_type: DocType
    chunk_index: int
    section_path: str = ""
    content: str


@lru_cache(maxsize=1)
def get_tokenizer():
    return AutoTokenizer.from_pretrained(TOKENIZER_MODEL)


def token_count(text: str) -> int:
    return len(get_tokenizer().encode(text, add_special_tokens=False))


def chunk_text(
    text: str,
    chunk_tokens: int = CHUNK_TOKENS,
    overlap_tokens: int = OVERLAP_TOKENS,
) -> list[str]:
    # Fixed token windows, sliced from the original text via character
    # offsets (decode(ids) would lowercase through the uncased tokenizer).
    tokenizer = get_tokenizer()
    offsets = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)[
        "offset_mapping"
    ]
    if not offsets:
        return []

    step = chunk_tokens - overlap_tokens
    chunks: list[str] = []
    start = 0
    while True:
        window = offsets[start : start + chunk_tokens]
        chunks.append(text[window[0][0] : window[-1][1]])
        if start + chunk_tokens >= len(offsets):
            break
        start += step
    return chunks


def _is_heading(line: str) -> bool:
    stripped = line.strip()
    if not (3 <= len(stripped) <= 60) or NOISE_RE.match(stripped):
        return False
    return (
        stripped == stripped.upper()
        and any(ch.isalpha() for ch in stripped)
        and len(stripped.split()) <= 8
    )


Section = tuple[str, str]  # (section_path, content)


def split_policy(text: str) -> list[Section]:
    """Split an ISO policy on PART headings and sections within each part."""
    part = ""
    section = ""
    buf: list[str] = []
    sections: list[Section] = []

    def flush() -> None:
        content = "\n".join(buf).strip()
        buf.clear()
        if not content:
            return
        path = " > ".join(p for p in (part, section) if p)
        if section == "DEFINITIONS":
            sections.extend(_split_definitions(content, path))
        else:
            sections.append((path, content))

    for line in text.splitlines():
        stripped = line.strip()
        if PART_RE.match(stripped) and _is_heading(stripped):
            flush()
            part, section = stripped, ""
        elif _is_heading(stripped):
            flush()
            section = stripped
        else:
            buf.append(line)
    flush()
    return sections


def _split_definitions(content: str, path: str) -> list[Section]:
    """Each lettered definition becomes its own chunk, named by its term."""
    items: list[list[str]] = [[]]
    for line in content.splitlines():
        if DEFINITION_ITEM_RE.match(line.strip()):
            items.append([line])
        else:
            items[-1].append(line)

    sections: list[Section] = []
    for item in items:
        item_text = "\n".join(item).strip()
        if not item_text:
            continue
        term = QUOTED_TERM_RE.search(item_text)
        label = f"{path} > {term.group(1)}" if term else path
        sections.append((label, item_text))
    return sections


def split_headed_doc(text: str, target_tokens: int) -> list[Section]:
    """Guides/statutes: group paragraphs under detected headings, then pack
    consecutive paragraphs up to the target size."""
    heading = ""
    packed: list[Section] = []
    buf: list[str] = []
    buf_tokens = 0

    def flush() -> None:
        nonlocal buf_tokens
        if buf:
            packed.append((heading, "\n\n".join(buf)))
        buf.clear()
        buf_tokens = 0

    for paragraph in re.split(r"\n\s*\n", text):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        first_line = paragraph.splitlines()[0]
        if _is_heading(first_line):
            flush()
            heading = first_line.strip()
        n = token_count(paragraph)
        if buf and buf_tokens + n > target_tokens:
            flush()
        buf.append(paragraph)
        buf_tokens += n
    flush()
    return packed


def _common_prefix(a: str, b: str) -> str:
    parts_a, parts_b = a.split(" > "), b.split(" > ")
    common = []
    for x, y in zip(parts_a, parts_b, strict=False):
        if x != y:
            break
        common.append(x)
    return " > ".join(common)


def pack_sections(sections: list[Section], target_tokens: int) -> list[Section]:
    """Greedily merge consecutive sibling sections (same top-level part) so
    tiny fragments — individual definitions, short provisions — don't become
    isolated low-signal chunks."""
    packed: list[Section] = []
    for path, content in sections:
        if packed:
            prev_path, prev_content = packed[-1]
            same_top = prev_path.split(" > ")[0] == path.split(" > ")[0]
            merged = prev_content + "\n\n" + content
            if same_top and token_count(merged) <= target_tokens:
                packed[-1] = (_common_prefix(prev_path, path), merged)
                continue
        packed.append((path, content))
    return packed


# Measured in Phase 3 and REVERTED: structure-aware chunking (both raw
# sections and sibling-packed) regressed recall@5 0.769 -> 0.615 and MRR
# 0.571 -> 0.435-0.466 against fixed 512/64 windows with bge embeddings.
# See NOTES/phase3.md and the v3-structure-chunks* rows in eval/RESULTS.md.
# The splitter is kept for reproducibility; flip to re-run the experiment.
STRUCTURE_CHUNKING = False


def chunk_document(doc: ParsedDocument) -> list[Chunk]:
    if not STRUCTURE_CHUNKING:
        return [
            Chunk(
                doc_name=doc.doc_name,
                state=doc.state,
                doc_type=doc.doc_type,
                chunk_index=i,
                section_path="",
                content=content,
            )
            for i, content in enumerate(chunk_text(doc.text))
        ]

    if doc.doc_type == "policy" and "PART A" in doc.text:
        sections = pack_sections(split_policy(doc.text), CHUNK_TOKENS)
    else:
        sections = split_headed_doc(doc.text, CHUNK_TOKENS)

    chunks: list[Chunk] = []
    for path, content in sections:
        body = f"[{path}]\n{content}" if path else content
        if token_count(body) <= CHUNK_TOKENS:
            pieces = [body]
        else:
            prefix = f"[{path}]\n" if path else ""
            # Leave headroom for the prefix so the embedder (512-token max)
            # never truncates the window.
            window = CHUNK_TOKENS - max(32, token_count(prefix))
            pieces = [prefix + piece for piece in chunk_text(content, chunk_tokens=window)]
        for piece in pieces:
            chunks.append(
                Chunk(
                    doc_name=doc.doc_name,
                    state=doc.state,
                    doc_type=doc.doc_type,
                    chunk_index=len(chunks),
                    section_path=path,
                    content=piece,
                )
            )
    return chunks
