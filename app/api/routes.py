from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from app.core.config import Settings, get_settings
from app.domain.models import (
    AnswerResponse,
    CompanyScore,
    DocumentInput,
    HealthResponse,
    QuestionRequest,
    RankRequest,
)
from app.services.intelligence import IntelligenceService


router = APIRouter()


def get_service() -> IntelligenceService:
    from app.main import app

    return app.state.service


@router.get("/health", response_model=HealthResponse)
def health(settings: Settings = Depends(get_settings)) -> HealthResponse:
    return HealthResponse(
        status="ok",
        service=settings.app_name,
        environment=settings.app_environment,
        mode=settings.app_mode,
        llm_provider=get_service().llm.name,
        embedding_provider=get_service().index.embedder.name,
        langfuse_enabled=get_service().observability.enabled,
    )


@router.get("/api/v1/corpus")
def corpus(service: IntelligenceService = Depends(get_service)) -> dict[str, object]:
    return service.index.corpus_summary()


@router.post("/api/v1/documents", status_code=status.HTTP_201_CREATED)
def ingest_documents(
    documents: list[DocumentInput],
    service: IntelligenceService = Depends(get_service),
) -> dict[str, int]:
    if not documents:
        raise HTTPException(status_code=400, detail="At least one document is required")
    return service.ingest(documents)


@router.post("/api/v1/ask", response_model=AnswerResponse)
def ask(
    request: QuestionRequest,
    service: IntelligenceService = Depends(get_service),
) -> AnswerResponse:
    return service.ask(request.question, request.company, request.top_k)


@router.post("/api/v1/rank", response_model=list[CompanyScore])
def rank(
    request: RankRequest,
    service: IntelligenceService = Depends(get_service),
) -> list[CompanyScore]:
    return service.rank(request.focus, request.companies)
