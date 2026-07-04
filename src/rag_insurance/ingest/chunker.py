"""Fixed-size token chunking — Phase 1 is intentionally naive.

512 tokens per chunk with 64 tokens of overlap, counted with the MiniLM
tokenizer so chunk sizes line up with what the embedder actually sees.
Structure-aware chunking is a Phase 3 measured change.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import BaseModel
from transformers import AutoTokenizer

from rag_insurance.ingest.parser import ParsedDocument
from rag_insurance.manifest import DocType

TOKENIZER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_TOKENS = 512
OVERLAP_TOKENS = 64


class Chunk(BaseModel):
    doc_name: str
    state: str
    doc_type: DocType
    chunk_index: int
    content: str


@lru_cache(maxsize=1)
def get_tokenizer():
    return AutoTokenizer.from_pretrained(TOKENIZER_MODEL)


def chunk_text(
    text: str,
    chunk_tokens: int = CHUNK_TOKENS,
    overlap_tokens: int = OVERLAP_TOKENS,
) -> list[str]:
    # Chunks are sliced from the original text via token character offsets
    # rather than decode(ids): decoding through the uncased MiniLM tokenizer
    # would lowercase content and drift at subword boundaries.
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


def chunk_document(doc: ParsedDocument) -> list[Chunk]:
    return [
        Chunk(
            doc_name=doc.doc_name,
            state=doc.state,
            doc_type=doc.doc_type,
            chunk_index=i,
            content=content,
        )
        for i, content in enumerate(chunk_text(doc.text))
    ]
