from app.rag.models import Document, RetrievalProfile, SearchHit
from app.rag.query.query_router import DefaultQueryRouter
from app.rag.query.reranker import LocalReranker
from app.rag.query.retrieval_fusion import ReciprocalRankFusion


def test_query_router_preserves_protocol_and_taxonomy_inference() -> None:
    router = DefaultQueryRouter()

    metadata_filter = router.infer_metadata_filter(
        "英威腾逆变器 0x3601 功能码 03",
        RetrievalProfile.PROTOCOL,
    )

    assert metadata_filter["document_type"] == "protocol"
    assert metadata_filter["device_type"] == "inverter"
    assert metadata_filter["manufacturer"] == "invt"
    assert metadata_filter["register_address"] == "0x3601"
    assert metadata_filter["function_code"] == "03"


def test_query_router_only_relaxes_manufacturer_and_deduplicates() -> None:
    router = DefaultQueryRouter()

    fallbacks = router.metadata_filter_fallbacks(
        {
            "document_type": "runbook",
            "device_type": "inverter",
            "manufacturer": "huawei",
        }
    )

    assert fallbacks == [
        {
            "document_type": "runbook",
            "device_type": "inverter",
            "manufacturer": "huawei",
        },
        {"document_type": "runbook", "device_type": "inverter"},
    ]


def test_reciprocal_rank_fusion_merges_duplicate_chunks() -> None:
    dense_document = Document("dense copy", {"_chunk_id": "shared"})
    sparse_document = Document("sparse copy", {"_chunk_id": "shared"})
    fusion = ReciprocalRankFusion()

    results = fusion.fuse(
        dense_results=[SearchHit(dense_document, 0.2, "dense")],
        sparse_results=[SearchHit(sparse_document, 2.0, "sparse")],
        k=60,
    )

    assert len(results) == 1
    assert results[0].source == "fused"
    assert results[0].document.page_content == "sparse copy"
    assert results[0].document.metadata["_retrieval_sources"] == ["dense", "sparse"]
    assert results[0].document.metadata["_retrieval_score_kind"] == "relevance"
    assert results[0].score == 2 / 61


def test_local_reranker_preserves_alert_and_runbook_boosts() -> None:
    ordinary = SearchHit(Document("general information"), 0.1, "dense")
    runbook = SearchHit(
        Document(
            "CPUHigh 排查 CPU 指标",
            {"alert_name": "CPUHigh", "runbook_section": "troubleshooting"},
        ),
        0.2,
        "dense",
    )

    ranked = LocalReranker().rerank(
        "CPUHigh 怎么排查",
        [ordinary, runbook],
        RetrievalProfile.AIOPS,
    )

    assert ranked[0].document is runbook.document
    assert ranked[0].document.metadata["_retrieval_profile"] == "aiops"
    assert ranked[0].document.metadata["_retrieval_score"] == ranked[0].score


def test_vector_relevance_score_keeps_relevance_and_transforms_distance_scores() -> None:
    reranker = LocalReranker()
    relevance_doc = Document("relevance hit", {"_retrieval_score_kind": "relevance"})
    distance_doc = Document("distance hit")

    assert reranker._vector_relevance_score(relevance_doc, 0.92) == 0.92
    assert reranker._vector_relevance_score(relevance_doc, -2.0) == 0.0
    assert reranker._vector_relevance_score(distance_doc, 4.0) == 1 / 5.0


def test_keyword_score_covers_document_fields_not_only_page_content() -> None:
    reranker = LocalReranker()
    hit_doc = Document(
        "普通说明",
        {
            "alert_name": "CPUHigh",
            "device_type": "inverter",
            "_file_name": "runbook-cpu.md",
        },
    )

    assert reranker._keyword_score("CPUHigh", hit_doc) == 1.0


def test_metadata_boost_combines_alert_runbook_taxonomy_protocol_and_headings() -> None:
    reranker = LocalReranker()
    hit_doc = Document(
        "查询 CPU 指标和告警排查",
        {
            "alert_name": "CPUTempHigh",
            "runbook_section": "troubleshooting",
            "device_type": "inverter",
            "manufacturer": "huawei",
            "register_address": "0x3601",
            "function_code": "03",
            "h1": "CPU temp high",
        },
    )

    score = reranker._metadata_boost(
        "CPUTempHigh 告警，华为逆变器 0x3601 功能码03",
        hit_doc,
        RetrievalProfile.AIOPS,
    )

    assert score == 1.0


def test_local_reranker_handles_scoring_exception_with_vector_fallback(monkeypatch) -> None:
    reranker = LocalReranker()

    def fail_keywords(*_: object, **__: object) -> float:
        raise RuntimeError("keyword explode")

    hit = SearchHit(
        Document("告警排查文档", {"_retrieval_score_kind": "distance"}),
        3.0,
        "dense",
    )
    monkeypatch.setattr(reranker, "_keyword_score", fail_keywords)
    ranked = reranker.rerank("CPUTempHigh", [hit], RetrievalProfile.AIOPS)

    assert ranked[0].score == 1 / (1 + 3.0)
    assert ranked[0].document.metadata["_retrieval_profile"] == "aiops"
    assert ranked[0].document.metadata["_retrieval_score"] == ranked[0].score
