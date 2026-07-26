from fastapi.testclient import TestClient

from app.main import app


def test_health_and_corpus_are_ready() -> None:
    with TestClient(app) as client:
        health = client.get("/health")
        corpus = client.get("/api/v1/corpus")

    assert health.status_code == 200
    assert health.json()["mode"] == "demo"
    assert health.json()["llm_provider"] == "deterministic"
    assert corpus.status_code == 200
    assert corpus.json()["documents"] == 5
    assert corpus.json()["chunks"] >= 5


def test_question_returns_verified_evidence() -> None:
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/ask",
            json={
                "question": "Which company provides governance and evaluation controls?",
                "company": "Safeguard AI",
            },
        )

    payload = response.json()
    assert response.status_code == 200
    assert payload["provider"] == "deterministic"
    assert payload["citations"]
    assert all(item["company"] == "Safeguard AI" for item in payload["citations"])
    assert payload["evaluation"]["citation_validity"] == 1.0


def test_arabic_question_retrieves_arabic_evidence() -> None:
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/ask",
            json={
                "question": "ما هي منصة حوكمة الذكاء الاصطناعي؟",
                "company": "Northstar Cloud",
            },
        )

    assert response.status_code == 200
    assert response.json()["citations"]


def test_ranking_is_ordered() -> None:
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/rank",
            json={"focus": "governance evaluation monitoring"},
        )

    ranking = response.json()
    assert response.status_code == 200
    assert len(ranking) >= 2
    assert ranking[0]["score"] >= ranking[-1]["score"]


def test_document_ingestion_updates_corpus() -> None:
    with TestClient(app) as client:
        created = client.post(
            "/api/v1/documents",
            json=[
                {
                    "document_id": "atlas-agents",
                    "company": "Atlas",
                    "title": "Agent Platform",
                    "text": "Atlas builds durable workflow agents with human approval and complete audit history.",
                    "language": "en",
                }
            ],
        )
        corpus = client.get("/api/v1/corpus")

    assert created.status_code == 201
    assert created.json()["chunks"] == 1
    assert corpus.json()["documents"] == 6
