from rag_insurance.ingest.chunker import CHUNK_TOKENS, OVERLAP_TOKENS, chunk_text, get_tokenizer


def token_ids(text: str) -> list[int]:
    return get_tokenizer().encode(text, add_special_tokens=False)


def test_empty_text_yields_no_chunks():
    assert chunk_text("") == []


def test_short_text_is_single_chunk():
    chunks = chunk_text("Collision coverage pays for damage to your car.")
    assert len(chunks) == 1


def test_chunk_sizes_and_overlap():
    # ~3000 tokens of varied text
    words = [f"coverage{i} exclusion{i} premium{i}" for i in range(500)]
    text = " ".join(words)
    ids = token_ids(text)
    assert len(ids) > 2 * CHUNK_TOKENS

    chunks = chunk_text(text)
    assert len(chunks) > 2

    step = CHUNK_TOKENS - OVERLAP_TOKENS
    offsets = get_tokenizer()(text, add_special_tokens=False, return_offsets_mapping=True)[
        "offset_mapping"
    ]
    for i, chunk in enumerate(chunks):
        window = offsets[i * step : i * step + CHUNK_TOKENS]
        # Every chunk must be exactly the text span of its 512-token window
        # (the final window may be shorter), which also pins the 64-token
        # overlap between consecutive chunks.
        assert chunk == text[window[0][0] : window[-1][1]]
        assert len(token_ids(chunk)) <= CHUNK_TOKENS

    # Full coverage: last window must reach the end of the document.
    last_start = (len(chunks) - 1) * step
    assert last_start + CHUNK_TOKENS >= len(ids)
