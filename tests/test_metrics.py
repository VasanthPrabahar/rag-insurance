from rag_insurance.eval.metrics import mrr, precision_at_k, recall_at_k, relevance_flags
from rag_insurance.retrieval.dense import RetrievedChunk


def make_chunk(doc_name: str, content: str) -> RetrievedChunk:
    return RetrievedChunk(
        doc_name=doc_name, state="TX", doc_type="guide", chunk_index=0, content=content, score=0.5
    )


def test_relevance_requires_doc_and_hint():
    chunks = [
        make_chunk("Doc A", "the deductible is what you pay"),  # right doc, right hint
        make_chunk("Doc B", "the deductible is what you pay"),  # wrong doc
        make_chunk("Doc A", "something unrelated"),  # right doc, no hint
    ]
    assert relevance_flags(chunks, ["Doc A"], "the deductible") == [True, False, False]


def test_relevance_is_case_and_whitespace_insensitive():
    chunks = [make_chunk("Doc A", "Breakage\nof   Glass is covered")]
    assert relevance_flags(chunks, ["Doc A"], "breakage of glass") == [True]


def test_relevance_empty_for_out_of_scope():
    chunks = [make_chunk("Doc A", "anything")]
    assert relevance_flags(chunks, [], None) == [False]


def test_recall_at_k():
    assert recall_at_k([False, False, True], 3) == 1.0
    assert recall_at_k([False, False, True], 2) == 0.0
    assert recall_at_k([], 5) == 0.0


def test_precision_at_k():
    assert precision_at_k([True, False, True, False], 4) == 0.5
    assert precision_at_k([True, True], 2) == 1.0


def test_mrr():
    assert mrr([False, True, False]) == 0.5
    assert mrr([True]) == 1.0
    assert mrr([False, False]) == 0.0
