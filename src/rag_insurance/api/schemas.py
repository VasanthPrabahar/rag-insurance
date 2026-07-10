"""Request/response schemas for the API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class EngineOptions(BaseModel):
    # "pipeline" streams tokens; "langchain"/"agent" run synchronously and
    # emit only the final SSE event.
    mode: Literal["pipeline", "langchain", "agent"] = "pipeline"
    rewrite: bool = True
    rerank: bool = False
    dense_only: bool = False


class Filters(BaseModel):
    state: str | None = None  # overrides regex state detection when set


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    k: int = Field(default=5, ge=1, le=20)
    engine: EngineOptions = EngineOptions()
    filters: Filters = Filters()


class Citation(BaseModel):
    chunk_id: int  # 1-based id within the retrieved set, as cited
    doc_name: str
    chunk_index: int
    section_path: str = ""


class RetrievedChunkMeta(BaseModel):
    chunk_id: int
    doc_name: str
    chunk_index: int
    score: float


class LatencyBreakdown(BaseModel):
    expand_ms: float
    retrieve_ms: float
    generate_ms: float
    total_ms: float


class AskFinal(BaseModel):
    answer: str
    refused: bool
    forced_refusal: bool
    citations: list[Citation]
    retrieved: list[RetrievedChunkMeta]
    latency: LatencyBreakdown


class HealthResponse(BaseModel):
    status: str  # "ok" | "degraded"
    db: bool
    ollama: bool


class StatsResponse(BaseModel):
    documents: int
    chunks: int


class IngestResponse(BaseModel):
    documents: int
    chunks: int
