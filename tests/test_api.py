"""API tests. Ollama is always mocked — CI has none. DB comes from the
compose/CI service; the whole module skips if it's unreachable."""

import json

import pytest
from fastapi.testclient import TestClient

from rag_insurance.ingest import store


def db_reachable() -> bool:
    try:
        store.connect().close()
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not db_reachable(), reason="database unreachable")


@pytest.fixture(scope="module")
def client():
    from rag_insurance.api.app import app

    # Tests must not depend on a prior ingest (CI runs pytest first):
    # ensure the schema exists; empty tables are fine for these tests.
    with store.connect() as conn:
        store.init_schema(conn)

    with TestClient(app) as test_client:  # context manager runs lifespan
        yield test_client


CANNED_JSON = (
    '{"answer": "Comprehensive covers animal impacts [1].", "citations": [1], "refused": false}'  # noqa: E501
)


def fake_stream(prompt, **kwargs):
    # Emit in several pieces to exercise incremental answer extraction.
    for i in range(0, len(CANNED_JSON), 17):
        yield CANNED_JSON[i : i + 17]


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["db"] is True
    assert body["status"] in ("ok", "degraded")


def test_stats(client):
    response = client.get("/stats")
    assert response.status_code == 200
    body = response.json()
    assert body["chunks"] >= 0


def test_ask_happy_path(client, monkeypatch):
    monkeypatch.setattr("rag_insurance.api.app.rewrite_query", lambda q: None)
    monkeypatch.setattr("rag_insurance.generation.ollama_client.generate_stream", fake_stream)

    with client.stream("POST", "/ask", json={"question": "Does insurance cover a deer?"}) as r:
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        events = []
        current_event = None
        for line in r.iter_lines():
            if line.startswith("event: "):
                current_event = line.removeprefix("event: ")
            elif line.startswith("data: "):
                events.append((current_event, json.loads(line.removeprefix("data: "))))

    kinds = [kind for kind, _ in events]
    assert kinds[-1] == "final"
    tokens = "".join(data["text"] for kind, data in events if kind == "token")
    final = events[-1][1]
    if not final["forced_refusal"]:  # empty DB forces a refusal; skip text check
        assert tokens == "Comprehensive covers animal impacts [1]."
        assert final["answer"] == "Comprehensive covers animal impacts [1]."
    assert {"expand_ms", "retrieve_ms", "generate_ms", "total_ms"} <= final["latency"].keys()
    assert isinstance(final["retrieved"], list)


def test_ask_rejects_bad_payload(client):
    assert client.post("/ask", json={"question": ""}).status_code == 422
    assert client.post("/ask", json={"question": "hi", "k": 0}).status_code == 422
    assert client.post("/ask", json={}).status_code == 422
