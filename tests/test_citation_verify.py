from rag_insurance.generation.answer import REFUSAL, GroundedAnswer, verify


def test_valid_citations_pass_through():
    result = verify(GroundedAnswer(answer="Fact [1] and [3].", citations=[1, 3], refused=False), 5)
    assert result.citations == [1, 3]
    assert not result.refused and not result.forced_refusal
    assert result.dropped_citations == []


def test_hallucinated_ids_dropped():
    result = verify(GroundedAnswer(answer="Fact [1][9].", citations=[1, 9, 0], refused=False), 5)
    assert result.citations == [1]
    assert result.dropped_citations == [9, 0]
    assert not result.refused


def test_zero_valid_citations_forces_refusal():
    result = verify(GroundedAnswer(answer="Confident claim [7].", citations=[7], refused=False), 5)
    assert result.refused and result.forced_refusal
    assert result.answer == REFUSAL
    assert result.citations == []


def test_model_refusal_normalized():
    result = verify(GroundedAnswer(answer="some refusal prose", citations=[], refused=True), 5)
    assert result.refused and not result.forced_refusal
    assert result.answer == REFUSAL
