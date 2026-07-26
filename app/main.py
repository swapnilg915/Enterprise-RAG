from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes import router
from app.core.config import get_settings
from app.core.observability import Observability
from app.providers.embeddings import build_embedding_provider
from app.providers.llm import build_llm_provider
from app.retrieval.index import DocumentIndex
from app.services.intelligence import IntelligenceService


@asynccontextmanager
async def lifespan(application: FastAPI):
    settings = get_settings()
    embedder = build_embedding_provider(settings)
    index = DocumentIndex(settings, embedder)
    index.seed_from_directory(settings.document_directory)
    observability = Observability(settings)
    application.state.service = IntelligenceService(
        settings=settings,
        index=index,
        llm=build_llm_provider(settings),
        observability=observability,
    )
    yield
    observability.flush()
    index.client.close()


settings = get_settings()
app = FastAPI(
    title=settings.app_name,
    version="2.0.0",
    description="Multilingual competitive-intelligence RAG with verified citations.",
    lifespan=lifespan,
)
app.include_router(router)
