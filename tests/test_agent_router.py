"""Router classification and decomposition tests — Ollama fully mocked."""

import pytest

from rag_insurance.agent import graph


def mock_generate(response: str):
    def fake(prompt, **kwargs):
        return response

    return fake


@pytest.mark.parametrize(
    "llm_output, expected",
    [
        ('{"route": "policy_lookup"}', "policy_lookup"),
        ('{"route": "state_law"}', "state_law"),
        ('{"route": "multi_part"}', "multi_part"),
        ('{"route": "out_of_scope"}', "out_of_scope"),
        # prose-wrapped JSON still parses (outermost-brace extraction)
        ('Sure! {"route": "state_law"} hope that helps', "state_law"),
        # malformed / unknown labels fall back to the safest route
        ('{"route": "philosophy"}', "policy_lookup"),
        ("not json at all", "policy_lookup"),
    ],
)
def test_classify(monkeypatch, llm_output, expected):
    monkeypatch.setattr(graph.ollama_client, "generate", mock_generate(llm_output))
    assert graph.classify("any question") == expected


def test_classify_falls_back_when_ollama_down(monkeypatch):
    def boom(prompt, **kwargs):
        raise ConnectionError("no ollama")

    monkeypatch.setattr(graph.ollama_client, "generate", boom)
    assert graph.classify("any question") == "policy_lookup"


def test_decompose_caps_at_three(monkeypatch):
    output = '{"sub_questions": ["a?", "b?", "c?", "d?", ""]}'
    monkeypatch.setattr(graph.ollama_client, "generate", mock_generate(output))
    state = {"question": "big question", "k": 5}
    graph.decompose_node(state)
    assert state["sub_questions"] == ["a?", "b?", "c?"]


def test_decompose_failure_falls_back_to_original(monkeypatch):
    monkeypatch.setattr(graph.ollama_client, "generate", mock_generate("garbage"))
    state = {"question": "the original question", "k": 5}
    graph.decompose_node(state)
    assert state["sub_questions"] == ["the original question"]


def test_refuse_node_skips_retrieval_and_generation():
    state = {"question": "best health insurance?", "k": 5}
    graph.refuse_node(state)
    assert state["answer"].refused is True
    assert state["chunks"] == []
    assert state["node_log"][-1]["node"] == "refuse"
