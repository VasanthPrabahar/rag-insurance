"""Grounded answer generation: question + retrieved chunks -> answer.

No citations yet — that's Phase 4.
"""

from __future__ import annotations

from rag_insurance.generation import ollama_client
from rag_insurance.retrieval.dense import RetrievedChunk

REFUSAL = "I don't know based on the provided documents."

PROMPT_TEMPLATE = """\
You are an assistant answering questions about auto insurance using ONLY the
context below, which comes from insurance policies, consumer guides, and
regulations.

Rules:
- Answer using only information stated in the context.
- If the context does not contain the answer, reply exactly:
  "{refusal}"
- Do not use outside knowledge, even if you are confident.

Context:
{context}

Question: {question}

Answer:"""


def build_prompt(question: str, chunks: list[RetrievedChunk]) -> str:
    context = "\n\n".join(
        f"--- [{chunk.doc_name} | {chunk.state} | {chunk.doc_type} | chunk {chunk.chunk_index}]\n"
        f"{chunk.content}"
        for chunk in chunks
    )
    return PROMPT_TEMPLATE.format(refusal=REFUSAL, context=context, question=question)


def answer(question: str, chunks: list[RetrievedChunk]) -> str:
    return ollama_client.generate(build_prompt(question, chunks)).strip()
