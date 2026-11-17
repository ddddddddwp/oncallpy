from __future__ import annotations

from typing import Any

import pytest

import app.rag.bootstrap as bootstrap
from app.rag.models import Document, RetrievalProfile, SearchHit, SearchQuery
from app.rag.query.models import QueryRequest, RewriteMode


@pytest.fixture(autouse=True)
def clear_bootstrap_caches():
    cached_factories = (
        bootstrap.get_embedding_provider,
        bootstrap.get_vector_store,
        bootstrap.get_sparse_index,
        bootstrap.get_indexing_service,
        bootstrap.get_query_rewrite_model,
        bootstrap.get_query_engine,
        bootstrap.get_query_service,
    )
    for factory in cached_factories:
        factory.cache_clear()
    yield
    for factory in cached_factories:
        factory.cache_clear()


class FakeVectorStore:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int, dict[str, Any] | None]] = []

    def search(
        self,
        query: str,
        k: int,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[SearchHit]:
        self.calls.append((query, k, metadata_filter))
        return [SearchHit(Document("dense result"), 0.1, "dense")]

    def supports_metadata_filter(self) -> bool:
        return True


def test_indexing_service_defers_backends_for_invalid_directory(
    tmp_path,
    monkeypatch,
) -> None:
    calls: list[str] = []

    def unavailable_vector_store():
        calls.append("vector")
        raise RuntimeError("vector unavailable")

    def unavailable_sparse_index():
        calls.append("sparse")
        raise RuntimeError("sparse unavailable")

    monkeypatch.setattr(bootstrap, "get_vector_store", unavailable_vector_store)
    monkeypatch.setattr(bootstrap, "get_sparse_index", unavailable_sparse_index)

    service = bootstrap.get_indexing_service()
    result = service.index_directory(str(tmp_path / "missing"))

    assert result.success is False
    assert "目录不存在" in result.error_message
    assert calls == []


def test_chat_query_does_not_initialize_sparse_index(monkeypatch) -> None:
    vector_store = FakeVectorStore()
    calls: list[str] = []

    def vector_provider():
        calls.append("vector")
        return vector_store

    def unavailable_sparse_index():
        calls.append("sparse")
        raise RuntimeError("sparse unavailable")

    monkeypatch.setattr(bootstrap, "get_vector_store", vector_provider)
    monkeypatch.setattr(bootstrap, "get_sparse_index", unavailable_sparse_index)

    service = bootstrap.get_query_service()
    assert calls == []

    documents = service.retrieve(
        SearchQuery("ordinary question", profile=RetrievalProfile.CHAT, top_k=1)
    )

    assert [document.page_content for document in documents] == ["dense result"]
    assert calls == ["vector", "vector", "vector"]


def test_hybrid_query_falls_back_when_sparse_initialization_fails(monkeypatch) -> None:
    vector_store = FakeVectorStore()
    calls: list[str] = []

    def vector_provider():
        calls.append("vector")
        return vector_store

    def unavailable_sparse_index():
        calls.append("sparse")
        raise RuntimeError("sparse unavailable")

    monkeypatch.setattr(bootstrap, "get_vector_store", vector_provider)
    monkeypatch.setattr(bootstrap, "get_sparse_index", unavailable_sparse_index)
    monkeypatch.setattr(bootstrap.config, "rag_route_retrieval_enabled", False)
    monkeypatch.setattr(bootstrap.config, "rag_hybrid_enabled", True)
    monkeypatch.setattr(bootstrap.config, "rag_hybrid_profiles", "protocol")

    service = bootstrap.get_query_service()
    documents = service.retrieve(
        SearchQuery("0x3601", profile=RetrievalProfile.PROTOCOL, top_k=1)
    )

    assert [document.page_content for document in documents] == ["dense result"]
    assert "sparse" in calls


def test_engine_wiring_defers_all_providers_until_the_selected_stage(monkeypatch) -> None:
    calls: list[str] = []

    monkeypatch.setattr(bootstrap.config, "rag_query_llm_enabled", True)
    monkeypatch.setattr(bootstrap, "get_vector_store", lambda: calls.append("vector"))
    monkeypatch.setattr(bootstrap, "get_sparse_index", lambda: calls.append("sparse"))
    monkeypatch.setattr(bootstrap, "get_query_rewrite_model", lambda: calls.append("llm"))

    engine = bootstrap.get_query_engine()
    engine.analyze(QueryRequest("ordinary question"))
    engine.plan(QueryRequest("ordinary question", rewrite_mode=RewriteMode.OFF))

    assert calls == []


def test_explicit_llm_rewrite_resolves_rewrite_model_only_on_demand(monkeypatch) -> None:
    calls: list[str] = []

    class FakeModel:
        def generate(self, payload):  # type: ignore[no-untyped-def]
            calls.append("llm")
            return {
                "primary_text": payload["text"],
                "subqueries": [],
                "entity_candidates": [],
                "confidence": 1.0,
                "reason": "test",
            }

    monkeypatch.setattr(bootstrap.config, "rag_query_llm_enabled", True)
    monkeypatch.setattr(bootstrap, "get_query_rewrite_model", FakeModel)

    engine = bootstrap.get_query_engine()
    engine.plan(QueryRequest("0x3601", profile=RetrievalProfile.PROTOCOL, rewrite_mode="llm"))

    assert calls == ["llm"]


def test_query_service_reuses_cached_engine_and_configured_mode(monkeypatch) -> None:
    monkeypatch.setattr(bootstrap.config, "rag_query_rewrite_mode", "rule")

    service = bootstrap.get_query_service()

    assert service.engine is bootstrap.get_query_engine()
    assert service.rewrite_mode == "rule"
