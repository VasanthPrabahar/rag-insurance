"""Streamlit chat demo against the FastAPI /ask endpoint.

Run:  uv run --group demo streamlit run demo/app.py
Needs the API up (docker compose up, or uvicorn locally) and Ollama running.
"""

import json
import os

import httpx
import streamlit as st

API = os.environ.get("RAG_API", "http://localhost:8000")

st.set_page_config(page_title="Auto Insurance RAG", page_icon="🚗")
st.title("Auto Insurance RAG")
st.caption("Local, cited, eval-gated. Answers come only from the ingested corpus.")

with st.sidebar:
    engine = st.radio("Engine", ["pipeline", "agent"], horizontal=True)
    state = st.selectbox("State filter", ["auto-detect", "TX", "CA", "AZ", "NV", "VA"])
    st.caption(
        "pipeline: expand → hybrid retrieve → cited generation (streams). "
        "agent: LangGraph router — refuses out-of-scope in ~1s, decomposes "
        "multi-part questions."
    )

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        for source in message.get("sources", []):
            with st.expander(source["label"]):
                st.text(source["text"])


def sse_events(response):
    event = None
    for line in response.iter_lines():
        if line.startswith("event: "):
            event = line.removeprefix("event: ")
        elif line.startswith("data: "):
            yield event, json.loads(line.removeprefix("data: "))


if question := st.chat_input("Ask about auto insurance…"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    payload = {
        "question": question,
        "engine": {"mode": engine},
        "filters": {"state": None if state == "auto-detect" else state},
    }
    with st.chat_message("assistant"):
        placeholder = st.empty()
        streamed = ""
        final = None
        with httpx.stream("POST", f"{API}/ask", json=payload, timeout=300.0) as response:
            for event, data in sse_events(response):
                if event == "token":
                    streamed += data["text"]
                    placeholder.markdown(streamed + "▌")
                elif event == "final":
                    final = data

        answer = final["answer"] if final else streamed
        placeholder.markdown(answer)

        sources = []
        if final:
            by_id = {c["chunk_id"]: c for c in final["retrieved"]}
            for citation in final["citations"]:
                chunk = by_id.get(citation["chunk_id"], {})
                label = (
                    f"[{citation['chunk_id']}] {citation['doc_name']} "
                    f"› chunk {citation['chunk_index']}"
                )
                sources.append({"label": label, "text": chunk.get("content", "")})
            latency = final["latency"]
            st.caption(
                f"{engine} · {latency['total_ms'] / 1000:.1f}s "
                f"(retrieve {latency['retrieve_ms'] / 1000:.1f}s · "
                f"generate {latency['generate_ms'] / 1000:.1f}s)"
                + (" · refused" if final["refused"] else "")
            )
        for source in sources:
            with st.expander(source["label"]):
                st.text(source["text"])

    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
