"""Embeddings via sentence-transformers all-MiniLM-L6-v2 (384 dims).

Normalized embeddings so cosine similarity reduces to a dot product and
pgvector's <=> distances are well-behaved.

Caveat (documented, intentional for Phase 1): MiniLM's max sequence length
is 256 tokens, so our 512-token chunks are truncated at embed time — the
embedding only "sees" the first half of each chunk.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384


def detect_device() -> str:
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


@lru_cache(maxsize=1)
def get_model() -> SentenceTransformer:
    return SentenceTransformer(MODEL_NAME, device=detect_device())


def embed_texts(texts: list[str], batch_size: int = 64) -> np.ndarray:
    return get_model().encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=len(texts) > batch_size,
    )


def embed_query(query: str) -> list[float]:
    return embed_texts([query])[0].tolist()
