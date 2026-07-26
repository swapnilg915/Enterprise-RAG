from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")


def _read_json(name: str) -> dict[str, Any]:
    config_dir = Path(os.getenv("APP_CONFIG_DIR", ROOT / "configs"))
    with (config_dir / name).open(encoding="utf-8") as config_file:
        return json.load(config_file)


def _boolean(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    app_name: str
    app_environment: str
    app_host: str
    app_port: int
    app_mode: str
    log_level: str
    llm_provider: str
    llm_model: str
    llm_temperature: float
    llm_timeout_seconds: float
    reasoning_enabled: bool
    reasoning_model: str
    reasoning_effort: str
    embedding_provider: str
    embedding_model: str
    embedding_dimension: int
    openai_api_key: str
    openai_base_url: str
    ollama_base_url: str
    qdrant_url: str
    qdrant_api_key: str
    qdrant_path: str
    qdrant_collection: str
    retrieval_k: int
    minimum_relevance: float
    document_directory: Path
    langfuse_enabled: bool
    langfuse_public_key: str
    langfuse_secret_key: str
    langfuse_base_url: str
    langfuse_environment: str

    @property
    def is_demo(self) -> bool:
        return self.app_mode == "demo"


@lru_cache
def get_settings() -> Settings:
    environment = _read_json("environment.json")
    llm = _read_json("llm.json")
    data = _read_json("data.json")
    reasoning = llm["reasoning"]
    fallback = llm["fallback"]
    provider = os.getenv("LLM_PROVIDER", llm["provider"]).lower()
    default_model = fallback["model"] if provider == "ollama" else llm["model"]
    default_embedding_provider = "ollama" if provider == "ollama" else llm["embedding_provider"]
    default_embedding_model = (
        fallback["embedding_model"] if provider == "ollama" else llm["embedding_model"]
    )
    return Settings(
        app_name=environment["app_name"],
        app_environment=os.getenv("APP_ENVIRONMENT", environment["environment"]),
        app_host=os.getenv("APP_HOST", environment["host"]),
        app_port=int(os.getenv("APP_PORT", environment["port"])),
        app_mode=os.getenv("APP_MODE", environment["mode"]).lower(),
        log_level=os.getenv("LOG_LEVEL", environment["log_level"]),
        llm_provider=provider,
        llm_model=os.getenv("LLM_MODEL", default_model),
        llm_temperature=float(os.getenv("LLM_TEMPERATURE", llm["temperature"])),
        llm_timeout_seconds=float(os.getenv("LLM_TIMEOUT_SECONDS", llm["timeout_seconds"])),
        reasoning_enabled=_boolean("LLM_REASONING_ENABLED", reasoning["enabled"]),
        reasoning_model=os.getenv("LLM_REASONING_MODEL", reasoning["model"]),
        reasoning_effort=os.getenv("LLM_REASONING_EFFORT", reasoning["effort"]),
        embedding_provider=os.getenv("EMBEDDING_PROVIDER", default_embedding_provider).lower(),
        embedding_model=os.getenv("EMBEDDING_MODEL", default_embedding_model),
        embedding_dimension=int(os.getenv("EMBEDDING_DIMENSION", llm["embedding_dimension"])),
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        openai_base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", fallback["base_url"]),
        qdrant_url=os.getenv("QDRANT_URL", ""),
        qdrant_api_key=os.getenv("QDRANT_API_KEY", ""),
        qdrant_path=os.getenv("QDRANT_PATH", data["index_directory"]),
        qdrant_collection=os.getenv("QDRANT_COLLECTION", data["collection"]),
        retrieval_k=int(os.getenv("RETRIEVAL_K", data["retrieval_k"])),
        minimum_relevance=float(
            os.getenv("MINIMUM_RELEVANCE", data["minimum_relevance"])
        ),
        document_directory=ROOT / os.getenv("DOCUMENT_DIRECTORY", data["document_directory"]),
        langfuse_enabled=_boolean("LANGFUSE_ENABLED", False),
        langfuse_public_key=os.getenv("LANGFUSE_PUBLIC_KEY", ""),
        langfuse_secret_key=os.getenv("LANGFUSE_SECRET_KEY", ""),
        langfuse_base_url=os.getenv("LANGFUSE_BASE_URL", "https://cloud.langfuse.com"),
        langfuse_environment=os.getenv(
            "LANGFUSE_TRACING_ENVIRONMENT", environment["environment"]
        ),
    )
