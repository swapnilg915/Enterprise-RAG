import math

from app.providers.embeddings import DeterministicEmbeddingProvider


def test_deterministic_embeddings_are_stable_and_normalized() -> None:
    provider = DeterministicEmbeddingProvider(dimension=64)

    first, second = provider.embed(["governed enterprise AI", "governed enterprise AI"])

    assert first == second
    assert len(first) == 64
    assert math.isclose(sum(value * value for value in first), 1.0)
