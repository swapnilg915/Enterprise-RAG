from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod

import httpx
from openai import OpenAI
from pydantic import ValidationError

from app.core.config import Settings
from app.domain.models import AnswerPayload, Citation


SYSTEM_PROMPT = """You are a competitive-intelligence analyst.
Answer only from the supplied evidence. Every factual claim must be supported by a
document_id in citation_ids. If evidence is insufficient, say so. Return JSON with:
answer (string), citation_ids (array of strings), confidence (0-1), limitations (array).
Do not include markdown fences."""


class LLMProvider(ABC):
    name: str
    model: str

    @abstractmethod
    def answer(self, question: str, citations: list[Citation]) -> AnswerPayload:
        raise NotImplementedError


def _prompt(question: str, citations: list[Citation]) -> str:
    evidence = [
        {
            "document_id": citation.document_id,
            "company": citation.company,
            "title": citation.title,
            "text": citation.excerpt,
        }
        for citation in citations
    ]
    return f"Question: {question}\nEvidence:\n{json.dumps(evidence, ensure_ascii=False)}"


def _parse_payload(raw_text: str) -> AnswerPayload:
    cleaned = re.sub(r"^```(?:json)?|```$", "", raw_text.strip()).strip()
    try:
        return AnswerPayload.model_validate_json(cleaned)
    except ValidationError as exc:
        raise RuntimeError("Model returned an invalid answer schema") from exc


class DeterministicLLMProvider(LLMProvider):
    name = "deterministic"
    model = "grounded-template-v1"

    def answer(self, question: str, citations: list[Citation]) -> AnswerPayload:
        del question
        if not citations:
            return AnswerPayload(
                answer="No sufficiently relevant evidence was found.",
                citation_ids=[],
                confidence=0.0,
                limitations=["No matching documents were available."],
            )
        statements = [f"{item.company}: {item.excerpt}" for item in citations[:3]]
        return AnswerPayload(
            answer=" ".join(statements),
            citation_ids=[item.document_id for item in citations[:3]],
            confidence=min(0.95, sum(item.relevance for item in citations[:3]) / 3 + 0.3),
            limitations=["Generated in deterministic demo mode."],
        )


class OpenAILLMProvider(LLMProvider):
    name = "openai"

    def __init__(self, settings: Settings) -> None:
        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is required when LLM_PROVIDER=openai")
        self.settings = settings
        self.model = (
            settings.reasoning_model
            if settings.reasoning_enabled and settings.reasoning_model
            else settings.llm_model
        )
        self.client = OpenAI(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            timeout=settings.llm_timeout_seconds,
        )

    def answer(self, question: str, citations: list[Citation]) -> AnswerPayload:
        request: dict[str, object] = {
            "model": self.model,
            "instructions": SYSTEM_PROMPT,
            "input": _prompt(question, citations),
        }
        if self.settings.reasoning_enabled:
            request["reasoning"] = {"effort": self.settings.reasoning_effort}
        else:
            request["temperature"] = self.settings.llm_temperature
        response = self.client.responses.create(**request)
        return _parse_payload(response.output_text)


class OllamaLLMProvider(LLMProvider):
    name = "ollama"

    def __init__(self, settings: Settings) -> None:
        self.model = settings.llm_model
        self.base_url = settings.ollama_base_url.rstrip("/")
        self.temperature = settings.llm_temperature
        self.timeout = settings.llm_timeout_seconds

    def answer(self, question: str, citations: list[Citation]) -> AnswerPayload:
        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "stream": False,
                    "format": "json",
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": _prompt(question, citations)},
                    ],
                    "options": {"temperature": self.temperature},
                },
            )
            response.raise_for_status()
        return _parse_payload(response.json()["message"]["content"])


def build_llm_provider(settings: Settings) -> LLMProvider:
    if settings.is_demo:
        return DeterministicLLMProvider()
    if settings.llm_provider == "openai":
        return OpenAILLMProvider(settings)
    if settings.llm_provider == "ollama":
        return OllamaLLMProvider(settings)
    raise ValueError(f"Unsupported LLM provider: {settings.llm_provider}")
