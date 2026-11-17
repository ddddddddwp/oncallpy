from dataclasses import asdict
from typing import Any, get_type_hints

from app.rag.models import RetrievalProfile, SearchQuery
from app.rag.query.models import (
    QueryAnalysis,
    QueryPlan,
    QueryRequest,
    QueryTraceEvent,
    RewriteDecision,
    RewriteMode,
    RewrittenQuery,
)


def test_query_request_copies_explicit_filter_from_search_query() -> None:
    original_filter = {"device_type": "inverter"}
    request = QueryRequest.from_search_query(
        SearchQuery(
            text="逆变器怎么排查？",
            profile=RetrievalProfile.AIOPS,
            top_k=5,
            metadata_filter=original_filter,
        )
    )
    original_filter["manufacturer"] = "huawei"

    assert request.text == "逆变器怎么排查？"
    assert request.profile == RetrievalProfile.AIOPS
    assert request.top_k == 5
    assert request.metadata_filter == {"device_type": "inverter"}
    assert request.rewrite_mode == RewriteMode.AUTO


def test_query_request_preserves_an_explicit_empty_filter_from_search_query() -> None:
    request = QueryRequest.from_search_query(SearchQuery(text="查询", metadata_filter={}))

    assert request.metadata_filter == {}


def test_query_request_and_nested_values_are_serializable() -> None:
    request = QueryRequest(text="寄存器 40001 是什么？", profile="protocol")
    payload = asdict(request)

    assert payload["text"] == "寄存器 40001 是什么？"
    assert payload["conversation_context"] == ()


def test_query_analysis_uses_default_values_for_optional_analysis_details() -> None:
    analysis = QueryAnalysis(
        normalized_text="寄存器 40001 是什么？",
        profile=RetrievalProfile.PROTOCOL,
    )

    assert analysis.intents == ()
    assert analysis.entities == {}
    assert analysis.inferred_filter == {}
    assert analysis.complexity == "simple"
    assert analysis.confidence == 1.0
    assert analysis.context_dependent is False
    assert analysis.reasons == ()


def test_rewrite_values_use_defaults_after_required_fields() -> None:
    decision = RewriteDecision(use_llm=False)
    rewrite = RewrittenQuery(primary_text="寄存器 40001")
    event = QueryTraceEvent(stage="rewrite", status="started")

    assert decision.reasons == ()
    assert decision.fallback == "rule"
    assert rewrite.subqueries == ()
    assert rewrite.preserved_terms == ()
    assert rewrite.entity_candidates == ()
    assert rewrite.source == "rule"
    assert rewrite.confidence == 1.0
    assert rewrite.reason == ""
    assert event.duration_ms == 0.0
    assert event.data == {}


def test_pipeline_model_annotations_preserve_filter_and_entity_types() -> None:
    analysis_hints = get_type_hints(QueryAnalysis)
    plan_hints = get_type_hints(QueryPlan)

    assert analysis_hints["entities"] == dict[str, str]
    assert plan_hints["metadata_filter"] == dict[str, Any] | None
