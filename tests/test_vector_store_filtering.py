from __future__ import annotations

import importlib
import sys
from types import ModuleType

import pytest
from langchain_core.documents import Document

from app.rag.infrastructure.milvus_vector_store import MilvusVectorStore
from app.rag.models import Document as DomainDocument


def test_legacy_vector_store_class_uses_infrastructure_implementation(monkeypatch):
    vector_store_module = _fresh_vector_store_module(monkeypatch)

    assert issubclass(vector_store_module.VectorStoreManager, MilvusVectorStore)

class FakeMilvus:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


class NoopConnections:
    @staticmethod
    def connect(**kwargs):
        _ = kwargs


def _fresh_vector_store_module(monkeypatch):
    fake_embedding_module = ModuleType("app.rag.vector_embedding_service")
    fake_embedding_module.vector_embedding_service = object()
    monkeypatch.setitem(sys.modules, "app.rag.vector_embedding_service", fake_embedding_module)

    fake_milvus_module = ModuleType("langchain_milvus")
    fake_milvus_module.Milvus = FakeMilvus
    monkeypatch.setitem(sys.modules, "langchain_milvus", fake_milvus_module)

    fake_pymilvus_module = ModuleType("pymilvus")
    fake_pymilvus_module.connections = NoopConnections
    monkeypatch.setitem(sys.modules, "pymilvus", fake_pymilvus_module)

    sys.modules.pop("app.rag.vector_store_manager", None)
    return importlib.import_module("app.rag.vector_store_manager")


def test_vector_store_builds_metadata_filter_expression(monkeypatch):
    vector_store_module = _fresh_vector_store_module(monkeypatch)
    manager = vector_store_module.VectorStoreManager.__new__(vector_store_module.VectorStoreManager)

    expr = manager._metadata_filter_expr(
        {
            "document_type": "runbook",
            "runbook_section": ["resolution", "troubleshooting"],
            "alert_name": 'CPU"High',
            "business_domain": r"payments\core",
            "empty_value": None,
            "route_key": "",
        }
    )

    assert 'metadata["document_type"] == "runbook"' in expr
    assert 'metadata["runbook_section"] == "resolution"' in expr
    assert 'metadata["runbook_section"] == "troubleshooting"' in expr
    assert 'CPU\\"High' in expr
    assert r"payments\\core" in expr
    assert "empty_value" not in expr
    assert "route_key" not in expr
    assert "&&" in expr
    assert "||" in expr


def test_vector_store_rejects_unsafe_metadata_filter_keys(monkeypatch):
    vector_store_module = _fresh_vector_store_module(monkeypatch)
    manager = vector_store_module.VectorStoreManager.__new__(vector_store_module.VectorStoreManager)

    with pytest.raises(ValueError, match="unsafe metadata filter key"):
        manager._metadata_filter_expr(
            {
                "document_type": "runbook",
                'bad"] || metadata["document_type': "route_summary",
            }
        )


class FilterFailingStore:
    def __init__(self):
        self.calls = []

    def similarity_search_with_score(self, query, k, expr=None):
        self.calls.append((query, k, expr))
        if expr:
            raise RuntimeError("bad expr")
        return [(Document(page_content="fallback", metadata={}), 0.2)]


def test_vector_store_filtered_search_fails_closed_on_expression_error(monkeypatch):
    vector_store_module = _fresh_vector_store_module(monkeypatch)
    manager = vector_store_module.VectorStoreManager.__new__(vector_store_module.VectorStoreManager)
    manager.vector_store = FilterFailingStore()

    docs = manager.similarity_search_with_score(
        "CPUHigh",
        k=3,
        metadata_filter={"document_type": "runbook"},
    )

    assert docs == []
    assert manager.vector_store.calls == [("CPUHigh", 3, 'metadata["document_type"] == "runbook"')]


class RecordingStore:
    def __init__(self):
        self.calls = []

    def similarity_search_with_score(self, query, k, expr=None):
        self.calls.append((query, k, expr))
        return [(Document(page_content="unfiltered", metadata={}), 0.1)]


def test_vector_store_no_filter_preserves_unfiltered_search(monkeypatch):
    vector_store_module = _fresh_vector_store_module(monkeypatch)
    manager = vector_store_module.VectorStoreManager.__new__(vector_store_module.VectorStoreManager)
    manager.vector_store = RecordingStore()

    docs = manager.similarity_search_with_score("CPUHigh", k=2)

    assert docs[0][0].page_content == "unfiltered"
    assert manager.vector_store.calls == [("CPUHigh", 2, None)]

    hits = manager.search("CPUHigh", k=2)
    assert hits[0].source == "dense"
    assert isinstance(hits[0].document, DomainDocument)
    assert hits[0].document.page_content == "unfiltered"


def test_vector_store_explicit_filter_with_no_safe_clause_fails_closed(monkeypatch):
    vector_store_module = _fresh_vector_store_module(monkeypatch)
    manager = vector_store_module.VectorStoreManager.__new__(vector_store_module.VectorStoreManager)
    manager.vector_store = RecordingStore()

    docs = manager.similarity_search_with_score(
        "CPUHigh",
        k=2,
        metadata_filter={'bad"] || metadata["document_type': "runbook"},
    )

    assert docs == []
    assert manager.vector_store.calls == []


class ParamRecordingStore:
    def __init__(self):
        self.calls = []

    def similarity_search_with_score(self, query, k, expr=None, param=None):
        self.calls.append((query, k, expr, param))
        return [(Document(page_content="with param", metadata={}), 0.1)]


def test_vector_store_forwards_configured_search_params(monkeypatch):
    vector_store_module = _fresh_vector_store_module(monkeypatch)
    monkeypatch.setattr(vector_store_module.config, "rag_vector_metric_type", "L2")
    monkeypatch.setattr(vector_store_module.config, "rag_vector_nprobe", 16)
    manager = vector_store_module.VectorStoreManager.__new__(vector_store_module.VectorStoreManager)
    manager.vector_store = ParamRecordingStore()

    docs = manager.similarity_search_with_score(
        "CPUHigh",
        k=2,
        metadata_filter={"document_type": "runbook"},
    )

    assert docs[0][0].page_content == "with param"
    assert manager.vector_store.calls == [
        (
            "CPUHigh",
            2,
            'metadata["document_type"] == "runbook"',
            {"metric_type": "L2", "params": {"nprobe": 16}},
        )
    ]


def test_vector_store_selects_configured_milvus_database(monkeypatch):
    vector_store_module = _fresh_vector_store_module(monkeypatch)
    monkeypatch.setattr(vector_store_module.config, "milvus_database_name", "oncall_agent")

    manager = vector_store_module.VectorStoreManager()

    assert manager.vector_store.kwargs["connection_args"]["db_name"] == "oncall_agent"


def test_vector_store_preconnects_orm_alias_to_configured_database(monkeypatch):
    vector_store_module = _fresh_vector_store_module(monkeypatch)
    monkeypatch.setattr(vector_store_module.config, "milvus_host", "milvus.internal")
    monkeypatch.setattr(vector_store_module.config, "milvus_port", 19531)
    monkeypatch.setattr(vector_store_module.config, "milvus_database_name", "oncall_agent")
    events = []

    class RecordingConnections:
        @staticmethod
        def connect(**kwargs):
            events.append(("connect", kwargs))

    class RecordingMilvus(FakeMilvus):
        def __init__(self, *args, **kwargs):
            events.append(("milvus", kwargs))
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(vector_store_module, "connections", RecordingConnections, raising=False)
    monkeypatch.setattr(vector_store_module, "Milvus", RecordingMilvus)

    vector_store_module.VectorStoreManager()

    assert events[0] == (
        "connect",
        {
            "alias": "http://milvus.internal:19531",
            "host": "milvus.internal",
            "port": 19531,
            "db_name": "oncall_agent",
        },
    )
    assert events[1][0] == "milvus"


class LegacyCollection:
    def __init__(self):
        self.query_calls = []
        self.delete_calls = []

    def query(self, **kwargs):
        self.query_calls.append(kwargs)
        return [
            {"id": "legacy-1", "metadata": {"_source": "/uploads/cpu.md"}},
            {
                "id": "current-1",
                "metadata": {
                    "_source": "/uploads/cpu.md",
                    "_file_id": "file-id",
                    "_file_hash": "file-hash",
                },
            },
        ]

    def delete(self, expr):
        self.delete_calls.append(expr)
        return type("DeleteResult", (), {"delete_count": 2})()


class FailingQueryCollection:
    def query(self, **kwargs):
        raise RuntimeError("milvus query unavailable")


def _install_milvus_manager(monkeypatch, collection):
    fake_client_module = ModuleType("app.rag.milvus_client")
    fake_client_module.milvus_manager = type(
        "MilvusManager",
        (),
        {"get_collection": lambda self: collection},
    )()
    monkeypatch.setitem(sys.modules, "app.rag.milvus_client", fake_client_module)


def test_vector_store_captures_only_legacy_ids_for_source(monkeypatch):
    vector_store_module = _fresh_vector_store_module(monkeypatch)
    collection = LegacyCollection()
    _install_milvus_manager(monkeypatch, collection)
    manager = vector_store_module.VectorStoreManager.__new__(vector_store_module.VectorStoreManager)

    document_ids = manager.get_legacy_ids_by_source('/uploads/cpu"guide.md')

    assert document_ids == ["legacy-1"]
    assert collection.query_calls == [
        {
            "expr": 'metadata["_source"] == "/uploads/cpu\\"guide.md"',
            "output_fields": ["id", "metadata"],
        }
    ]


def test_vector_store_deletes_exact_primary_ids(monkeypatch):
    vector_store_module = _fresh_vector_store_module(monkeypatch)
    collection = LegacyCollection()
    _install_milvus_manager(monkeypatch, collection)
    manager = vector_store_module.VectorStoreManager.__new__(vector_store_module.VectorStoreManager)

    deleted = manager.delete_by_ids(["legacy-1", 'legacy-2"quoted'])

    assert deleted == 2
    assert collection.delete_calls == ['id in ["legacy-1", "legacy-2\\"quoted"]']


def test_vector_store_has_file_hash_propagates_query_failure(monkeypatch):
    vector_store_module = _fresh_vector_store_module(monkeypatch)
    _install_milvus_manager(monkeypatch, FailingQueryCollection())
    manager = vector_store_module.VectorStoreManager.__new__(vector_store_module.VectorStoreManager)

    with pytest.raises(RuntimeError, match="milvus query unavailable"):
        manager.has_file_hash("file-id", "file-hash")


def test_vector_store_has_file_id_propagates_query_failure(monkeypatch):
    vector_store_module = _fresh_vector_store_module(monkeypatch)
    _install_milvus_manager(monkeypatch, FailingQueryCollection())
    manager = vector_store_module.VectorStoreManager.__new__(vector_store_module.VectorStoreManager)

    with pytest.raises(RuntimeError, match="milvus query unavailable"):
        manager.has_file_id("file-id")
