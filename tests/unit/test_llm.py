from app.domain.models import Citation
from app.providers.llm import DeterministicLLMProvider


def test_deterministic_provider_returns_only_supplied_citations() -> None:
    citation = Citation(
        document_id="doc-1",
        company="Example",
        title="AI Strategy",
        excerpt="Example operates a governed AI platform.",
        relevance=0.8,
    )

    result = DeterministicLLMProvider().answer("What is the strategy?", [citation])

    assert result.citation_ids == ["doc-1"]
    assert result.confidence > 0
