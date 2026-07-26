from __future__ import annotations

import json
import re
import uuid
from pathlib import Path

from llama_index.core.node_parser import SentenceSplitter
from qdrant_client import QdrantClient, models

from app.core.config import Settings
from app.domain.models import Citation, DocumentInput, IndexedDocument
from app.providers.embeddings import EmbeddingProvider


def _terms(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"\w+", text.lower(), flags=re.UNICODE)
        if len(token) > 2
    }


class DocumentIndex:
    def __init__(self, settings: Settings, embedder: EmbeddingProvider) -> None:
        self.settings = settings
        self.embedder = embedder
        self.collection = f"{settings.qdrant_collection}_{embedder.name}_{embedder.dimension}"
        if settings.is_demo:
            self.client = QdrantClient(":memory:")
        elif settings.qdrant_url:
            self.client = QdrantClient(
                url=settings.qdrant_url,
                api_key=settings.qdrant_api_key or None,
            )
        else:
            self.client = QdrantClient(path=settings.qdrant_path)
        self.splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        collections = {
            item.name for item in self.client.get_collections().collections
        }
        if self.collection not in collections:
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=models.VectorParams(
                    size=self.embedder.dimension,
                    distance=models.Distance.COSINE,
                ),
            )

    def count(self) -> int:
        return self.client.count(collection_name=self.collection, exact=True).count

    def ingest(self, documents: list[DocumentInput]) -> int:
        texts: list[str] = []
        payloads: list[dict[str, object]] = []
        point_ids: list[str] = []
        for document in documents:
            indexed = IndexedDocument(**document.model_dump())
            chunks = self.splitter.split_text(indexed.text)
            for position, chunk in enumerate(chunks):
                chunk_id = f"{indexed.document_id}-chunk-{position + 1}"
                texts.append(chunk)
                payloads.append(
                    {
                        **indexed.model_dump(),
                        "document_id": chunk_id,
                        "parent_document_id": indexed.document_id,
                        "text": chunk,
                        "chunk_position": position,
                    }
                )
                point_ids.append(str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id)))
        if not texts:
            return 0
        vectors = self.embedder.embed(texts)
        self.client.upsert(
            collection_name=self.collection,
            wait=True,
            points=[
                models.PointStruct(id=point_id, vector=vector, payload=payload)
                for point_id, vector, payload in zip(point_ids, vectors, payloads, strict=True)
            ],
        )
        return len(texts)

    def seed_from_directory(self, directory: Path) -> int:
        if self.count() > 0:
            return 0
        documents: list[DocumentInput] = []
        for path in sorted(directory.glob("*.json")):
            with path.open(encoding="utf-8") as document_file:
                payload = json.load(document_file)
            items = payload if isinstance(payload, list) else [payload]
            documents.extend(DocumentInput.model_validate(item) for item in items)
        return self.ingest(documents)

    def retrieve(
        self,
        question: str,
        company: str | None,
        limit: int,
    ) -> list[Citation]:
        query_vector = self.embedder.embed([question])[0]
        query_filter = None
        if company:
            query_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="company",
                        match=models.MatchValue(value=company),
                    )
                ]
            )
        result = self.client.query_points(
            collection_name=self.collection,
            query=query_vector,
            query_filter=query_filter,
            limit=limit,
            with_payload=True,
        )
        question_terms = _terms(question)
        citations: list[Citation] = []
        for point in result.points:
            payload = point.payload or {}
            lexical = len(question_terms & _terms(str(payload.get("text", "")))) / max(
                len(question_terms), 1
            )
            vector_score = max(0.0, min(1.0, float(point.score)))
            relevance = round(vector_score * 0.7 + lexical * 0.3, 4)
            if relevance < self.settings.minimum_relevance:
                continue
            citations.append(
                Citation(
                    document_id=str(payload["document_id"]),
                    company=str(payload["company"]),
                    title=str(payload["title"]),
                    excerpt=str(payload["text"]),
                    source_url=(
                        str(payload["source_url"]) if payload.get("source_url") else None
                    ),
                    relevance=relevance,
                )
            )
        return citations

    def corpus_summary(self) -> dict[str, object]:
        records, _ = self.client.scroll(
            collection_name=self.collection,
            limit=10_000,
            with_payload=True,
            with_vectors=False,
        )
        companies: dict[str, int] = {}
        parents: set[str] = set()
        for record in records:
            payload = record.payload or {}
            company = str(payload.get("company", "unknown"))
            companies[company] = companies.get(company, 0) + 1
            parents.add(str(payload.get("parent_document_id", record.id)))
        return {
            "documents": len(parents),
            "chunks": len(records),
            "companies": dict(sorted(companies.items())),
            "collection": self.collection,
        }
