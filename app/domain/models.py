from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field


class DocumentInput(BaseModel):
    document_id: str = Field(pattern=r"^[a-zA-Z0-9._-]{1,100}$")
    company: str = Field(min_length=1, max_length=100)
    title: str = Field(min_length=1, max_length=300)
    text: str = Field(min_length=20, max_length=100_000)
    language: str = Field(default="en", pattern=r"^[a-z]{2}$")
    source_url: str | None = None
    published_at: str | None = None


class IndexedDocument(DocumentInput):
    indexed_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class QuestionRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2_000)
    company: str | None = Field(default=None, max_length=100)
    top_k: int | None = Field(default=None, ge=1, le=20)


class Citation(BaseModel):
    document_id: str
    company: str
    title: str
    excerpt: str
    source_url: str | None = None
    relevance: float = Field(ge=0, le=1)


class AnswerPayload(BaseModel):
    answer: str
    citation_ids: list[str]
    confidence: float = Field(ge=0, le=1)
    limitations: list[str] = Field(default_factory=list)


class EvaluationScores(BaseModel):
    citation_validity: float = Field(ge=0, le=1)
    retrieval_relevance: float = Field(ge=0, le=1)
    answer_coverage: float = Field(ge=0, le=1)


class AnswerResponse(BaseModel):
    answer: str
    citations: list[Citation]
    evaluation: EvaluationScores
    provider: str
    model: str
    trace_id: str | None = None


class RankRequest(BaseModel):
    focus: str = Field(min_length=3, max_length=1_000)
    companies: list[str] | None = Field(default=None, max_length=20)


class CompanyScore(BaseModel):
    company: str
    score: int = Field(ge=0, le=100)
    matching_themes: list[str]
    evidence_ids: list[str]


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    service: str
    environment: str
    mode: str
    llm_provider: str
    embedding_provider: str
    langfuse_enabled: bool
