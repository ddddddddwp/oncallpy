from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from app.rag.models import Document, SearchHit
from app.rag.ports import (
    EmbeddingProvider,
    SparseSearcher,
    SparseWriter,
    VectorSearcher,
    VectorWriter,
)


class FakeVectorStore:
    def __init__(self) -> None:
        self.documents: list[Document] = []

    def add_documents(self, documents: Sequence[Document]) -> list[str]:
        self.documents.extend(documents)
        return [str(index) for index, _document in enumerate(documents)]

    def has_file_id(self, file_id: str) -> bool:
        return file_id == "file-id"

    def has_file_hash(self, file_id: str, file_hash: str) -> bool:
        return (file_id, file_hash) == ("file-id", "file-hash")

    def get_legacy_ids_by_source(self, source: str) -> list[str]:
        return ["legacy-id"] if source == "source.md" else []

    def delete_by_ids(self, document_ids: list[str]) -> int:
        return len(document_ids)

    def delete_by_file_generation(self, file_id: str, file_hash: str) -> int:
        return int(bool(file_id and file_hash))

    def delete_file_generations(self, file_id: str, keep_file_hash: str) -> int:
        return int(bool(file_id and keep_file_hash))

    def optimize_collection(self) -> dict[str, Any]:
        return {"optimized": True}

    def search(
        self,
        query: str,
        k: int,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[SearchHit]:
        del metadata_filter
        return [SearchHit(Document(query), 0.1, "dense")][:k]

    def supports_metadata_filter(self) -> bool:
        return True


class FakeSparseIndex:
    def __init__(self) -> None:
        self.documents: list[Document] = []

    def upsert_documents(self, documents: Sequence[Document]) -> None:
        self.documents.extend(documents)

    def replace_file_documents(self, file_id: str, documents: Sequence[Document]) -> int:
        del file_id
        self.documents = list(documents)
        return 0

    def contains_file_hash(self, file_id: str, file_hash: str) -> bool:
        return (file_id, file_hash) == ("file-id", "file-hash")

    def search_hits(
        self,
        query: str,
        k: int,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[SearchHit]:
        del metadata_filter
        return [SearchHit(Document(query), 1.0, "sparse")][:k]


class FakeEmbeddings:
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(text))] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return [float(len(text))]


vector_writer: VectorWriter = FakeVectorStore()
vector_searcher: VectorSearcher = FakeVectorStore()
sparse_writer: SparseWriter = FakeSparseIndex()
sparse_searcher: SparseSearcher = FakeSparseIndex()
embedding_provider: EmbeddingProvider = FakeEmbeddings()


def test_vector_ports_are_satisfied_structurally() -> None:
    documents = [Document("content")]

    assert vector_writer.add_documents(documents) == ["0"]
    assert vector_writer.has_file_hash("file-id", "file-hash") is True
    assert vector_searcher.search("query", 1)[0].source == "dense"
    assert vector_searcher.supports_metadata_filter() is True


def test_sparse_ports_are_satisfied_structurally() -> None:
    documents = [Document("content")]

    sparse_writer.upsert_documents(documents)

    assert sparse_writer.contains_file_hash("file-id", "file-hash") is True
    assert sparse_searcher.search_hits("query", 1)[0].source == "sparse"


def test_embedding_provider_is_satisfied_structurally() -> None:
    assert embedding_provider.embed_documents(["one", "three"]) == [[3.0], [5.0]]
    assert embedding_provider.embed_query("query") == [5.0]
