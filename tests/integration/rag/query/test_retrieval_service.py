from __future__ import annotations

import importlib
import inspect
import sys
from contextlib import contextmanager
from typing import Any

from langchain_core.documents import Document

import app.rag.query.query_service as query_service_module
from app.rag.models import (
    Document as DomainDocument,
    RetrievalProfile,
    SearchHit,
    SearchQuery,
)
from app.rag.query.query_router import DefaultQueryRouter
from app.rag.query.query_service import QueryService
from app.rag.query.reranker import LocalReranker
from app.rag.query.retrieval_fusion import ReciprocalRankFusion


def _to_domain_document(document: Any) -> DomainDocument:
    return DomainDocument(
        page_content=str(document.page_content),
        metadata=dict(document.metadata or {}),
        id=getattr(document, "id", None),
    )


class VectorSearcherAdapter:
    def __init__(self, store: Any) -> None:
        self.store = store

    def search(
        self,
        query: str,
        k: int,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[SearchHit]:
        if metadata_filter is None:
            candidates = self.store.similarity_search_with_score(query, k=k)
        else:
            candidates = self.store.similarity_search_with_score(
                query,
                k=k,
                metadata_filter=metadata_filter,
            )
        return [
            SearchHit(_to_domain_document(document), float(score), "dense")
            for document, score in candidates
        ]

    def supports_metadata_filter(self) -> bool:
        method = getattr(self.store, "similarity_search_with_score", None)
        if method is None:
            return False
        try:
            signature = inspect.signature(method)
        except (TypeError, ValueError):
            return True
        return "metadata_filter" in signature.parameters or any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        )


class SparseSearcherAdapter:
    def __init__(self, store: Any) -> None:
        self.store = store

    def search_hits(
        self,
        query: str,
        k: int,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[SearchHit]:
        return [
            SearchHit(_to_domain_document(document), float(score), "sparse")
            for document, score in self.store.search(
                query,
                k=k,
                metadata_filter=metadata_filter,
            )
        ]


def make_query_service(
    vector_store: Any,
    sparse_search: Any | None = None,
    trace_span_factory: Any | None = None,
) -> QueryService:
    kwargs = {}
    if trace_span_factory is not None:
        kwargs["trace_span_factory"] = trace_span_factory
    return QueryService(
        vector_searcher=VectorSearcherAdapter(vector_store),
        sparse_searcher=(
            SparseSearcherAdapter(sparse_search) if sparse_search is not None else None
        ),
        router=DefaultQueryRouter(),
        fusion=ReciprocalRankFusion(),
        reranker=LocalReranker(),
        **kwargs,
    )


class DomainVectorSearcher:
    def search(self, query, k, metadata_filter=None):
        return [
            SearchHit(
                DomainDocument(
                    "CPUHigh 排查步骤",
                    {
                        "alert_name": "CPUHigh",
                        "runbook_section": "troubleshooting",
                    },
                ),
                0.2,
                "dense",
            )
        ][:k]

    def supports_metadata_filter(self):
        return True


def test_new_query_service_returns_domain_documents() -> None:
    service = QueryService(
        vector_searcher=DomainVectorSearcher(),
        sparse_searcher=None,
        router=DefaultQueryRouter(),
        fusion=ReciprocalRankFusion(),
        reranker=LocalReranker(),
    )

    documents = service.retrieve(
        SearchQuery(
            "CPUHigh 怎么排查",
            RetrievalProfile.AIOPS,
            1,
            {"alert_name": "CPUHigh"},
        )
    )

    assert len(documents) == 1
    assert isinstance(documents[0], DomainDocument)
    assert documents[0].page_content == "CPUHigh 排查步骤"


class FakeStore:
    def __init__(self):
        self.calls = []

    def similarity_search_with_score(self, query, k, metadata_filter=None):
        self.calls.append((query, k))
        return [
            (
                Document(
                    page_content="普通说明",
                    metadata={"runbook_section": "other"},
                ),
                0.1,
            ),
            (
                Document(
                    page_content="CPUHigh 排查步骤: 查询 CPU 指标",
                    metadata={"alert_name": "CPUHigh", "runbook_section": "troubleshooting"},
                ),
                0.4,
            ),
        ]


class FilterAwareStore:
    def __init__(self):
        self.calls = []

    def similarity_search_with_score(self, query, k, metadata_filter=None):
        self.calls.append((query, k, metadata_filter))
        return [
            (
                Document(
                    page_content="再查询 checkout 错误日志。",
                    metadata={
                        "_context_window": "先查看 CPU 指标。 再查询 checkout 错误日志。 最后确认突增流量。",
                        "_original_text": "再查询 checkout 错误日志。",
                        "document_type": "runbook",
                        "alert_name": "CPUHigh",
                        "runbook_section": "troubleshooting",
                    },
                ),
                0.2,
            )
        ]


class CompatibilityStoreWithoutFilter:
    def __init__(self):
        self.calls = []

    def similarity_search_with_score(self, query, k):
        self.calls.append((query, k))
        return [(Document(page_content="unfiltered leak", metadata={}), 0.1)]


class InferredFilterStore:
    def __init__(self):
        self.calls = []

    def similarity_search_with_score(self, query, k, metadata_filter=None):
        self.calls.append((query, k, metadata_filter))
        return [
            (
                Document(
                    page_content="CPUHigh 排查步骤",
                    metadata={"document_type": "runbook", "runbook_section": "troubleshooting"},
                ),
                0.2,
            )
        ]


class TaxonomyFilterStore:
    def __init__(self):
        self.calls = []

    def similarity_search_with_score(self, query, k, metadata_filter=None):
        self.calls.append((query, k, metadata_filter))
        return [
            (
                Document(
                    page_content="华为逆变器 PVOverVoltage 处理步骤",
                    metadata={
                        "document_type": "runbook",
                        "runbook_section": "resolution",
                        "device_type": "inverter",
                        "manufacturer": "huawei",
                    },
                ),
                0.2,
            )
        ]


class ChatTaxonomyFilterStore:
    def __init__(self):
        self.calls = []

    def similarity_search_with_score(self, query, k, metadata_filter=None):
        self.calls.append((query, k, metadata_filter))
        if metadata_filter == {"device_type": "inverter", "manufacturer": "invt"}:
            return [
                (
                    Document(
                        page_content="英威腾逆变器 0x3601 告警信息",
                        metadata={
                            "document_type": "document",
                            "device_type": "inverter",
                            "manufacturer": "invt",
                        },
                    ),
                    0.2,
                )
            ]
        return []


class ProtocolFilterStore:
    def __init__(self):
        self.calls = []

    def similarity_search_with_score(self, query, k, metadata_filter=None):
        self.calls.append((query, k, metadata_filter))
        if metadata_filter == {
            "document_type": "protocol",
            "device_type": "inverter",
            "manufacturer": "invt",
            "register_address": "0x3601",
        }:
            return [
                (
                    Document(
                        page_content="0x3601 告警状态 功能码 03 uint16",
                        metadata={
                            "document_type": "protocol",
                            "device_type": "inverter",
                            "manufacturer": "invt",
                            "register_address": "0x3601",
                            "register_name": "告警状态",
                        },
                    ),
                    0.1,
                )
            ]
        return []


class StrictTaxonomyMissStore:
    def __init__(self):
        self.calls = []

    def similarity_search_with_score(self, query, k, metadata_filter=None):
        self.calls.append((query, k, metadata_filter))
        if metadata_filter is None:
            return [
                (
                    Document(
                        page_content="BST 逆变器通用告警",
                        metadata={"_file_name": "BST逆变模块modbus协议V0.10__241225.md"},
                    ),
                    0.2,
                )
            ]
        return []


class SparseSearchStore:
    def __init__(self):
        self.calls = []

    def search(self, query, k, metadata_filter=None):
        self.calls.append((query, k, metadata_filter))
        return [
            (
                Document(
                    page_content="0x3601 告警状态 功能码 03 uint16",
                    metadata={
                        "_chunk_id": "protocol-sparse",
                        "document_type": "protocol",
                        "register_address": "0x3601",
                    },
                ),
                4.0,
            )
        ]


class DenseMissStore:
    def __init__(self):
        self.calls = []

    def similarity_search_with_score(self, query, k, metadata_filter=None):
        self.calls.append((query, k, metadata_filter))
        return [
            (
                Document(
                    page_content="逆变器常规说明",
                    metadata={"_chunk_id": "dense-general", "document_type": "protocol"},
                ),
                0.2,
            )
        ]


class HybridRankStore:
    def similarity_search_with_score(self, query, k, metadata_filter=None):
        return [
            (
                Document(
                    page_content="shared protocol term",
                    metadata={"_chunk_id": "rank-a", "document_type": "protocol"},
                ),
                0.1,
            ),
            (
                Document(
                    page_content="shared protocol term",
                    metadata={"_chunk_id": "rank-b", "document_type": "protocol"},
                ),
                0.2,
            ),
        ]


class HybridRankSparseStore:
    def search(self, query, k, metadata_filter=None):
        return [
            (
                Document(
                    page_content="shared protocol term",
                    metadata={"_chunk_id": "rank-a", "document_type": "protocol"},
                ),
                5.0,
            ),
            (
                Document(
                    page_content="shared protocol term",
                    metadata={"_chunk_id": "rank-b", "document_type": "protocol"},
                ),
                4.0,
            ),
        ]


class TaxonomyFallbackStore:
    def __init__(self):
        self.calls = []

    def similarity_search_with_score(self, query, k, metadata_filter=None):
        self.calls.append((query, k, metadata_filter))
        if metadata_filter and metadata_filter.get("manufacturer") == "huawei":
            return []
        if metadata_filter and metadata_filter.get("device_type") == "inverter":
            return [
                (
                    Document(
                        page_content="逆变器通用 PVOverVoltage 处理步骤",
                        metadata={
                            "document_type": "runbook",
                            "runbook_section": "resolution",
                            "device_type": "inverter",
                        },
                    ),
                    0.3,
                )
            ]
        return []


def test_retrieval_passes_explicit_metadata_filter_and_replaces_context():
    store = FilterAwareStore()
    service = make_query_service(store)

    docs = service.retrieve(
        SearchQuery(
            "CPUHigh 怎么排查",
            profile=RetrievalProfile.AIOPS,
            top_k=1,
            metadata_filter={
                "alert_name": "CPUHigh",
                "runbook_section": "troubleshooting",
            },
        )
    )

    assert store.calls[0][2] == {"alert_name": "CPUHigh", "runbook_section": "troubleshooting"}
    assert isinstance(docs[0], DomainDocument)
    assert docs[0].page_content == "先查看 CPU 指标。 再查询 checkout 错误日志。 最后确认突增流量。"
    assert docs[0].metadata["_original_text"] == "再查询 checkout 错误日志。"


def test_explicit_filter_fails_closed_for_compatibility_store_type_error():
    store = CompatibilityStoreWithoutFilter()
    service = make_query_service(store)

    docs = service.retrieve(
        SearchQuery(
            "CPUHigh",
            profile=RetrievalProfile.CHAT,
            top_k=1,
            metadata_filter={"alert_name": "CPUHigh"},
        )
    )

    assert docs == []
    assert store.calls == []


def test_aiops_query_infers_runbook_section_filter(monkeypatch):
    monkeypatch.setattr(query_service_module.config, "rag_route_retrieval_enabled", False)
    store = InferredFilterStore()
    service = make_query_service(store)

    docs = service.retrieve(
        SearchQuery("CPUHigh 怎么排查", profile=RetrievalProfile.AIOPS, top_k=1)
    )

    assert docs
    assert store.calls[0][2] == {
        "document_type": "runbook",
        "runbook_section": "troubleshooting",
    }


def test_aiops_query_infers_device_type_and_manufacturer_filter(monkeypatch):
    monkeypatch.setattr(query_service_module.config, "rag_route_retrieval_enabled", False)
    store = TaxonomyFilterStore()
    service = make_query_service(store)

    docs = service.retrieve(
        SearchQuery(
            "华为逆变器 PVOverVoltage 怎么处理",
            profile=RetrievalProfile.AIOPS,
            top_k=1,
        )
    )

    assert docs
    assert store.calls[0][2] == {
        "document_type": "runbook",
        "runbook_section": "resolution",
        "device_type": "inverter",
        "manufacturer": "huawei",
    }


def test_chat_query_infers_device_type_and_manufacturer_filter(monkeypatch):
    monkeypatch.setattr(query_service_module.config, "rag_route_retrieval_enabled", False)
    store = ChatTaxonomyFilterStore()
    service = make_query_service(store)

    docs = service.retrieve(
        SearchQuery(
            "英威腾逆变器 0x3601 告警",
            profile=RetrievalProfile.CHAT,
            top_k=1,
        )
    )

    assert docs
    assert store.calls[0][2] == {
        "device_type": "inverter",
        "manufacturer": "invt",
    }


def test_protocol_query_infers_register_address_filter(monkeypatch):
    monkeypatch.setattr(query_service_module.config, "rag_route_retrieval_enabled", False)
    store = ProtocolFilterStore()
    service = make_query_service(store)

    docs = service.retrieve(
        SearchQuery(
            "英威腾逆变器 0x3601 寄存器含义",
            profile=RetrievalProfile.PROTOCOL,
            top_k=1,
        )
    )

    assert docs
    assert store.calls[0][2] == {
        "document_type": "protocol",
        "device_type": "inverter",
        "manufacturer": "invt",
        "register_address": "0x3601",
    }


def test_protocol_retrieval_uses_sparse_candidates_for_exact_terms(monkeypatch):
    monkeypatch.setattr(query_service_module.config, "rag_route_retrieval_enabled", False)
    monkeypatch.setattr(query_service_module.config, "rag_hybrid_enabled", True, raising=False)
    monkeypatch.setattr(
        query_service_module.config,
        "rag_hybrid_profiles",
        "aiops,protocol",
        raising=False,
    )
    dense_store = DenseMissStore()
    sparse_store = SparseSearchStore()
    service = make_query_service(dense_store, sparse_store)

    docs = service.retrieve(
        SearchQuery(
            "英威腾 0x3601 功能码 03",
            profile=RetrievalProfile.PROTOCOL,
            top_k=1,
        )
    )

    assert docs[0].metadata["_chunk_id"] == "protocol-sparse"
    assert dense_store.calls
    assert sparse_store.calls
    assert sparse_store.calls[0][2]["document_type"] == "protocol"


def test_hybrid_rrf_relevance_order_survives_rerank(monkeypatch):
    monkeypatch.setattr(query_service_module.config, "rag_route_retrieval_enabled", False)
    monkeypatch.setattr(query_service_module.config, "rag_hybrid_enabled", True, raising=False)
    monkeypatch.setattr(
        query_service_module.config, "rag_hybrid_profiles", "protocol", raising=False
    )
    service = make_query_service(
        HybridRankStore(),
        HybridRankSparseStore(),
    )

    docs = service.retrieve(SearchQuery("shared", profile=RetrievalProfile.PROTOCOL, top_k=2))

    assert [doc.metadata["_chunk_id"] for doc in docs] == ["rank-a", "rank-b"]
    assert all(doc.metadata["_retrieval_score_kind"] == "relevance" for doc in docs)


def test_taxonomy_query_does_not_fall_back_to_unfiltered_when_filter_misses(monkeypatch):
    monkeypatch.setattr(query_service_module.config, "rag_route_retrieval_enabled", False)
    store = StrictTaxonomyMissStore()
    service = make_query_service(store)

    docs = service.retrieve(
        SearchQuery(
            "英威腾逆变器 0x3601 告警",
            profile=RetrievalProfile.CHAT,
            top_k=1,
        )
    )

    assert docs == []
    assert all(call[2] is not None for call in store.calls)


def test_aiops_taxonomy_filter_relaxes_to_device_type_when_manufacturer_empty(
    monkeypatch,
):
    monkeypatch.setattr(query_service_module.config, "rag_route_retrieval_enabled", False)
    store = TaxonomyFallbackStore()
    service = make_query_service(store)

    docs = service.retrieve(
        SearchQuery(
            "华为逆变器 PVOverVoltage 怎么处理",
            profile=RetrievalProfile.AIOPS,
            top_k=1,
        )
    )

    assert docs[0].metadata["device_type"] == "inverter"
    assert store.calls[0][2] == {
        "document_type": "runbook",
        "runbook_section": "resolution",
        "device_type": "inverter",
        "manufacturer": "huawei",
    }
    assert store.calls[1][2] == {
        "document_type": "runbook",
        "runbook_section": "resolution",
        "device_type": "inverter",
    }


class RouteAwareStore:
    def __init__(self, routed_docs=None):
        self.calls = []
        self.routed_docs = routed_docs or [
            (
                Document(
                    page_content="CPUHigh route",
                    metadata={
                        "document_type": "route_summary",
                        "alert_name": "CPUHigh",
                        "target_document_type": "runbook",
                    },
                ),
                0.1,
            ),
            (
                Document(
                    page_content="CPU content",
                    metadata={"document_type": "runbook", "alert_name": "CPUHigh"},
                ),
                0.2,
            ),
        ]

    def similarity_search_with_score(self, query, k, metadata_filter=None):
        self.calls.append((query, k, metadata_filter))
        if metadata_filter and metadata_filter.get("document_type") == "route_summary":
            return [self.routed_docs[0]]
        return [self.routed_docs[1]]


class RouteTaxonomySearchStore:
    def __init__(self):
        self.calls = []

    def similarity_search_with_score(self, query, k, metadata_filter=None):
        self.calls.append((query, k, metadata_filter))
        if metadata_filter == {
            "document_type": "route_summary",
            "device_type": "inverter",
            "manufacturer": "invt",
        }:
            return [
                (
                    Document(
                        page_content="英威腾逆变器 route",
                        metadata={
                            "document_type": "route_summary",
                            "target_document_type": "document",
                            "device_type": "inverter",
                            "manufacturer": "invt",
                            "route_key": "inverter/invt/工作表：新列表",
                            "_source": "/uploads/英威腾逆变器寄存器协议.md",
                        },
                    ),
                    0.1,
                )
            ]
        if (
            metadata_filter
            and metadata_filter.get("_source") == "/uploads/英威腾逆变器寄存器协议.md"
        ):
            return [
                (
                    Document(
                        page_content="英威腾逆变器 0x3601 告警信息",
                        metadata={
                            "document_type": "document",
                            "device_type": "inverter",
                            "manufacturer": "invt",
                        },
                    ),
                    0.2,
                )
            ]
        return []


def test_retrieval_routes_through_summary_then_content_filter():
    store = RouteAwareStore()
    service = make_query_service(store)

    docs = service.retrieve(
        SearchQuery("CPUHigh 怎么处理", profile=RetrievalProfile.AIOPS, top_k=1)
    )

    assert docs
    assert store.calls[0][2] == {"document_type": "route_summary"}
    assert store.calls[1][2]["document_type"] == "runbook"
    assert store.calls[1][2]["alert_name"] == "CPUHigh"


def test_route_summary_search_uses_query_taxonomy_filter():
    store = RouteTaxonomySearchStore()
    service = make_query_service(store)

    docs = service.retrieve(
        SearchQuery(
            "英威腾逆变器 0x3601 告警",
            profile=RetrievalProfile.CHAT,
            top_k=1,
        )
    )

    assert docs
    assert store.calls[0][2] == {
        "document_type": "route_summary",
        "device_type": "inverter",
        "manufacturer": "invt",
    }
    assert store.calls[1][2]["_source"] == "/uploads/英威腾逆变器寄存器协议.md"


def test_retrieval_route_filter_includes_device_taxonomy():
    route_doc = Document(
        page_content="华为逆变器 PVOverVoltage route",
        metadata={
            "document_type": "route_summary",
            "target_document_type": "runbook",
            "alert_name": "PVOverVoltage",
            "device_type": "inverter",
            "manufacturer": "huawei",
        },
    )
    content_doc = Document(
        page_content="华为逆变器 PVOverVoltage 处理步骤",
        metadata={"document_type": "runbook", "alert_name": "PVOverVoltage"},
    )
    store = RouteAwareStore(routed_docs=[(route_doc, 0.1), (content_doc, 0.2)])
    service = make_query_service(store)

    docs = service.retrieve(
        SearchQuery(
            "华为逆变器 PVOverVoltage 怎么处理",
            profile=RetrievalProfile.AIOPS,
            top_k=1,
        )
    )

    assert docs
    assert store.calls[1][2] == {
        "document_type": "runbook",
        "alert_name": "PVOverVoltage",
        "device_type": "inverter",
        "manufacturer": "huawei",
    }


class EmptyRouteStore:
    def __init__(self):
        self.calls = []

    def similarity_search_with_score(self, query, k, metadata_filter=None):
        self.calls.append((query, k, metadata_filter))
        if metadata_filter == {"document_type": "route_summary"}:
            return []
        return [(Document(page_content="broad fallback", metadata={}), 0.3)]


def test_retrieval_falls_back_when_route_missing():
    store = EmptyRouteStore()
    service = make_query_service(store)

    docs = service.retrieve(SearchQuery("unknown alert", profile=RetrievalProfile.AIOPS, top_k=1))

    assert docs[0].page_content == "broad fallback"
    assert store.calls[0][2] == {"document_type": "route_summary"}
    assert store.calls[-1][2] != {"document_type": "route_summary"}


def test_aiops_retrieval_reranks_alert_runbook_chunks_first():
    service = make_query_service(FakeStore())

    docs = service.retrieve(
        SearchQuery("CPUHigh 怎么排查", profile=RetrievalProfile.AIOPS, top_k=1)
    )

    assert docs[0].metadata["alert_name"] == "CPUHigh"


def test_unknown_profile_falls_back_to_chat():
    service = make_query_service(FakeStore())

    docs = service.retrieve(SearchQuery("CPUHigh", profile="unknown", top_k=1))

    assert len(docs) == 1


def test_aiops_profile_uses_larger_candidate_recall():
    store = FakeStore()
    service = make_query_service(store)

    service.retrieve(SearchQuery("CPUHigh", profile=RetrievalProfile.AIOPS, top_k=5))

    assert ("CPUHigh", 15) in store.calls


def test_retrieval_reports_trace_span(monkeypatch):
    spans = []

    @contextmanager
    def fake_trace_span(name, span_type="custom", input_data=None, set_output=None):
        spans.append(
            {
                "name": name,
                "span_type": span_type,
                "input": input_data,
                "set_output": set_output,
                "manual_output": None,
            }
        )

        class FakeSpan:
            def set_output(self, output):
                spans[-1]["manual_output"] = output

        yield FakeSpan()

    service = make_query_service(FakeStore(), trace_span_factory=fake_trace_span)

    docs = service.retrieve(SearchQuery("CPUHigh", profile=RetrievalProfile.AIOPS, top_k=1))

    assert docs
    assert spans[0]["name"] == "rag_retrieve"
    assert spans[0]["input"] == {"query": "CPUHigh", "profile": "aiops", "top_k": 1}
    assert spans[0]["manual_output"] == {
        "status": "success",
        "documents": 1,
        "candidate_k": 15,
        "metadata_filter": {"document_type": "runbook"},
        "route_attempted": True,
        "route_matched": False,
        "fallback_reason": "route_miss",
    }


def test_knowledge_tool_passes_profile_to_query_service(monkeypatch):
    sys.modules.pop("app.tools.knowledge_tool", None)
    knowledge_module = importlib.import_module("app.tools.knowledge_tool")
    calls: list[SearchQuery] = []

    class FakeQueryService:
        def retrieve(self, query: SearchQuery) -> list[DomainDocument]:
            calls.append(query)
            return [
                DomainDocument(
                    page_content="CPUHigh 排查步骤",
                    metadata={"_file_name": "cpu.md"},
                    id="cpu-chunk",
                )
            ]

    monkeypatch.setattr(
        knowledge_module,
        "get_query_service",
        lambda: FakeQueryService(),
    )

    context, docs = knowledge_module.retrieve_knowledge.func("CPUHigh", profile="aiops", top_k=7)

    assert calls == [SearchQuery(text="CPUHigh", profile="aiops", top_k=7)]
    assert "CPUHigh 排查步骤" in context
    assert len(docs) == 1
    assert isinstance(docs[0], Document)
    assert docs[0].id == "cpu-chunk"


def test_knowledge_tool_keeps_facade_signature_and_forwards_service_rewrite_mode(monkeypatch):
    """The tool remains a LangChain artifact tool while QueryService owns defaults."""
    sys.modules.pop("app.tools.knowledge_tool", None)
    knowledge_module = importlib.import_module("app.tools.knowledge_tool")

    class CapturingEngine:
        request = None

        def query(self, request):
            self.request = request
            plan = query_service_module.DefaultQueryPlanner(
                query_service_module._planner_settings()
            ).plan(
                request,
                query_service_module.DefaultQueryAnalyzer().analyze(request),
                query_service_module.DefaultRewritePolicy(
                    query_service_module.RewritePolicySettings(llm_enabled=False)
                ).decide(request, query_service_module.DefaultQueryAnalyzer().analyze(request)),
                query_service_module.RuleQueryRewriter().rewrite(
                    request, query_service_module.DefaultQueryAnalyzer().analyze(request)
                ),
            )
            from app.rag.query.models import QueryResult

            return QueryResult([DomainDocument("body", {"nested": {"a": 1}})], plan)

    @contextmanager
    def noop_span(*args, **kwargs):
        class Span:
            def set_output(self, output):
                return None

        yield Span()

    engine = CapturingEngine()
    monkeypatch.setattr(
        knowledge_module,
        "get_query_service",
        lambda: QueryService(engine=engine, trace_span_factory=noop_span),
    )
    assert inspect.signature(knowledge_module.retrieve_knowledge.func).parameters["profile"].default == "chat"
    assert inspect.signature(knowledge_module.retrieve_knowledge.func).parameters["top_k"].default is None
    context, documents = knowledge_module.retrieve_knowledge.func("hello", profile="protocol", top_k=2)
    assert context and engine.request.profile == "protocol"
    assert engine.request.top_k == 2 and engine.request.rewrite_mode == "off"
    assert documents[0].metadata == {"nested": {"a": 1}}
