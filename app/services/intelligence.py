from __future__ import annotations

import re

from app.core.config import Settings
from app.core.observability import Observability
from app.domain.models import (
    AnswerResponse,
    CompanyScore,
    DocumentInput,
    EvaluationScores,
)
from app.providers.llm import LLMProvider
from app.retrieval.index import DocumentIndex


def _terms(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"\w+", text.lower(), flags=re.UNICODE)
        if len(token) > 2
    }


class IntelligenceService:
    def __init__(
        self,
        settings: Settings,
        index: DocumentIndex,
        llm: LLMProvider,
        observability: Observability,
    ) -> None:
        self.settings = settings
        self.index = index
        self.llm = llm
        self.observability = observability

    def ingest(self, documents: list[DocumentInput]) -> dict[str, int]:
        with self.observability.span(
            "document-ingestion",
            input_data={"documents": len(documents)},
        ):
            chunks = self.index.ingest(documents)
        return {"documents": len(documents), "chunks": chunks}

    def ask(
        self,
        question: str,
        company: str | None,
        top_k: int | None,
    ) -> AnswerResponse:
        limit = top_k or self.settings.retrieval_k
        with self.observability.span(
            "answer-question",
            input_data={"question": question, "company": company, "top_k": limit},
            metadata={"provider": self.llm.name, "model": self.llm.model},
        ) as observation:
            citations = self.index.retrieve(question, company, limit)
            payload = self.llm.answer(question, citations)
            valid_ids = {citation.document_id for citation in citations}
            selected = [
                citation
                for citation in citations
                if citation.document_id in payload.citation_ids
            ]
            invalid_ids = set(payload.citation_ids) - valid_ids
            citation_failure = bool(invalid_ids) or (
                bool(citations) and not payload.citation_ids
            )
            question_terms = _terms(question)
            answer_text = (
                "The generated response failed citation validation and was withheld."
                if citation_failure
                else payload.answer
            )
            if citation_failure:
                selected = []
            answer_terms = _terms(answer_text)
            evaluation = EvaluationScores(
                citation_validity=(
                    1.0
                    if not payload.citation_ids and not citations
                    else round(
                        (len(payload.citation_ids) - len(invalid_ids))
                        / max(len(payload.citation_ids), 1),
                        4,
                    )
                ),
                retrieval_relevance=round(
                    sum(item.relevance for item in citations)
                    / max(len(citations), 1),
                    4,
                ),
                answer_coverage=round(
                    len(question_terms & answer_terms) / max(len(question_terms), 1),
                    4,
                ),
            )
            trace_id = getattr(observation, "trace_id", None)
            response = AnswerResponse(
                answer=answer_text,
                citations=selected,
                evaluation=evaluation,
                provider=self.llm.name,
                model=self.llm.model,
                trace_id=trace_id,
            )
            if observation is not None and hasattr(observation, "update"):
                observation.update(output=response.model_dump())
            return response

    def rank(
        self,
        focus: str,
        companies: list[str] | None,
    ) -> list[CompanyScore]:
        requested = companies or []
        citations = self.index.retrieve(focus, None, 100)
        grouped: dict[str, list] = {}
        for citation in citations:
            if requested and citation.company not in requested:
                continue
            grouped.setdefault(citation.company, []).append(citation)
        focus_terms = _terms(focus)
        ranking: list[CompanyScore] = []
        for company, evidence in grouped.items():
            evidence_terms = _terms(" ".join(item.excerpt for item in evidence))
            themes = sorted(focus_terms & evidence_terms)
            mean_relevance = sum(item.relevance for item in evidence) / len(evidence)
            score = min(100, round(mean_relevance * 80 + min(len(themes), 4) * 5))
            ranking.append(
                CompanyScore(
                    company=company,
                    score=score,
                    matching_themes=themes,
                    evidence_ids=[item.document_id for item in evidence[:5]],
                )
            )
        return sorted(ranking, key=lambda item: (-item.score, item.company))
