"""Extract plain text from the raw corpus, with manifest metadata attached.

PDFs go through pymupdf, HTML through BeautifulSoup. Headers/footers are
stripped heuristically: a line that opens or closes most pages of a PDF
(page numbers, running titles) is noise, not content.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pymupdf
from bs4 import BeautifulSoup
from pydantic import BaseModel

from rag_insurance.manifest import BY_FILENAME, DocType

# A line is treated as a running header/footer if it appears at the edge of
# at least this fraction of pages (only applied to docs with enough pages
# for the frequency signal to mean anything).
EDGE_LINES = 2
MIN_PAGES_FOR_STRIPPING = 4
EDGE_FREQUENCY_THRESHOLD = 0.5


class ParsedDocument(BaseModel):
    doc_name: str
    state: str
    doc_type: DocType
    filename: str
    text: str


def _normalize_line(line: str) -> str:
    # Collapse digits so "Page 3 of 14" and "Page 4 of 14" count as the same
    # recurring header line.
    return "".join("#" if ch.isdigit() else ch for ch in line.strip().lower())


def _strip_headers_footers(pages: list[list[str]]) -> str:
    if len(pages) < MIN_PAGES_FOR_STRIPPING:
        return "\n".join(line for page in pages for line in page)

    edge_counts: Counter[str] = Counter()
    for page in pages:
        edges = page[:EDGE_LINES] + page[-EDGE_LINES:]
        for line in set(_normalize_line(ln) for ln in edges if ln.strip()):
            edge_counts[line] += 1

    threshold = len(pages) * EDGE_FREQUENCY_THRESHOLD
    noise = {line for line, count in edge_counts.items() if count >= threshold}

    kept: list[str] = []
    for page in pages:
        for i, line in enumerate(page):
            is_edge = i < EDGE_LINES or i >= len(page) - EDGE_LINES
            if is_edge and _normalize_line(line) in noise:
                continue
            kept.append(line)
    return "\n".join(kept)


def parse_pdf(path: Path) -> str:
    with pymupdf.open(path) as doc:
        pages = [page.get_text().splitlines() for page in doc]
    return _strip_headers_footers(pages)


def parse_html(path: Path) -> str:
    soup = BeautifulSoup(path.read_text(errors="replace"), "html.parser")
    for tag in soup(["script", "style", "nav", "header", "footer"]):
        tag.decompose()
    return soup.get_text(separator="\n")


def _clean(text: str) -> str:
    lines = [line.strip() for line in text.splitlines()]
    out: list[str] = []
    blank = False
    for line in lines:
        if not line:
            if not blank:
                out.append("")
            blank = True
        else:
            out.append(line)
            blank = False
    return "\n".join(out).strip()


def parse_corpus(data_dir: Path) -> list[ParsedDocument]:
    documents: list[ParsedDocument] = []
    for path in sorted(data_dir.iterdir()):
        if path.name.startswith("."):
            continue
        meta = BY_FILENAME.get(path.name)
        if meta is None:
            print(f"WARN    {path.name} not in manifest, skipping")
            continue
        if path.suffix.lower() == ".pdf":
            text = parse_pdf(path)
        elif path.suffix.lower() in (".html", ".htm"):
            text = parse_html(path)
        else:
            print(f"WARN    {path.name} has unsupported extension, skipping")
            continue
        documents.append(
            ParsedDocument(
                doc_name=meta.name,
                state=meta.state,
                doc_type=meta.doc_type,
                filename=meta.filename,
                text=_clean(text),
            )
        )
    return documents
