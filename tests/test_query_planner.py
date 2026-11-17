"""Tests for declarative, bounded retrieval planning."""

from __future__ import annotations

import pytest

from app.rag.models import RetrievalProfile, SearchQuery
from app.rag.query.analyzer import DefaultQueryAnalyzer
from app.rag.query.metadata_registry import MetadataFieldRegistry, MetadataFieldSpec
from app.rag.query.models import (
    EntityCandidate,
    QueryAnalysis,
    QueryRequest,
    RewriteDecision,
    RewrittenQuery,
)
from app.rag.query.planner import DefaultQueryPlanner, QueryPlannerSettings


def make_analysis(**overrides: object) -> QueryAnalysis:
    values: dict[str, object] = {
        "normalized_text": "华为逆变器告警排查",
        "profile": RetrievalProfile.CHAT,
        "intents": ("troubleshooting", "verification"),
        "inferred_filter": {"document_type": "runbook", "manufacturer": "huawei"},
    }
    values.update(overrides)
    return QueryAnalysis(**values)  # type: ignore[arg-type]


def make_rewrite(**overrides: object) -> RewrittenQuery:
    values: dict[str, object] = {"primary_text": "主查询", "source": "rule"}
    values.update(overrides)
    return RewrittenQuery(**values)  # type: ignore[arg-type]


def test_planner_bounds_deduplicates_queries_and_keeps_attempt_limit() -> None:
    plan = DefaultQueryPlanner(QueryPlannerSettings(max_subqueries=2, max_attempts=7)).plan(
        QueryRequest(text="原始", top_k=5),
        make_analysis(),
        RewriteDecision(use_llm=False),
        make_rewrite(subqueries=("", "主查询", "子查询", "第二子查询", "超量")),
    )

    assert plan.queries == ("主查询", "子查询", "第二子查询")
    assert plan.final_top_k == 5
    assert plan.max_attempts == 7
    assert plan.route_first is True


def test_explicit_filter_is_deep_copied_even_when_empty_and_disables_routing() -> None:
    source = {"model": {"names": ["SUN2000"]}}
    plan = DefaultQueryPlanner().plan(
        QueryRequest(text="查询", metadata_filter=source),
        make_analysis(),
        RewriteDecision(use_llm=False),
        make_rewrite(entity_candidates=(EntityCandidate("manufacturer", "huawei", "华为"),)),
    )
    source["model"]["names"].append("changed")

    assert plan.metadata_filter == {"model": {"names": ["SUN2000"]}}
    assert plan.route_first is False

    empty = DefaultQueryPlanner().plan(
        QueryRequest(text="查询", metadata_filter={}),
        make_analysis(),
        RewriteDecision(use_llm=False),
        make_rewrite(),
    )
    assert empty.metadata_filter == {}
    assert empty.route_first is False


def test_planner_uses_profile_defaults_and_normalizes_invalid_request_top_k() -> None:
    planner = DefaultQueryPlanner(QueryPlannerSettings(top_k=3, aiops_top_k=8))
    aiops = planner.plan(
        QueryRequest(text="查询", profile="aiops", top_k=0),
        make_analysis(profile=RetrievalProfile.AIOPS),
        RewriteDecision(use_llm=False),
        make_rewrite(),
    )
    chat = planner.plan(
        QueryRequest(text="查询", profile="chat", top_k=True),
        make_analysis(profile=RetrievalProfile.CHAT),
        RewriteDecision(use_llm=False),
        make_rewrite(),
    )

    assert (aiops.final_top_k, aiops.candidate_k) == (8, 24)
    assert (chat.final_top_k, chat.candidate_k) == (3, 6)


@pytest.mark.parametrize(
    ("profile", "top_k", "candidate_k"),
    [
        (RetrievalProfile.CHAT, None, 6),
        (RetrievalProfile.CHAT, 1, 6),
        (RetrievalProfile.AIOPS, None, 24),
        (RetrievalProfile.AIOPS, 1, 15),
        (RetrievalProfile.PROTOCOL, None, 15),
        (RetrievalProfile.PROTOCOL, 1, 15),
    ],
)
def test_planner_preserves_profile_candidate_depth_floors(
    profile: RetrievalProfile, top_k: int | None, candidate_k: int
) -> None:
    plan = DefaultQueryPlanner().plan(
        QueryRequest(text="查询", profile=profile, top_k=top_k),
        make_analysis(profile=profile),
        RewriteDecision(use_llm=False),
        make_rewrite(),
    )

    assert plan.candidate_k == candidate_k


def test_planner_bounds_route_and_sparse_depth_settings() -> None:
    plan = DefaultQueryPlanner(
        QueryPlannerSettings(max_top_k=10, route_top_k=999, sparse_top_k=False)  # type: ignore[arg-type]
    ).plan(
        QueryRequest(text="查询"),
        make_analysis(),
        RewriteDecision(use_llm=False),
        make_rewrite(),
    )

    assert plan.route_k == 30
    assert plan.sparse_min_k == 20


def test_planner_honors_switches_and_normalizes_integer_settings() -> None:
    settings = QueryPlannerSettings(
        route_enabled=False,
        hybrid_enabled=False,
        rrf_k=0,
        max_attempts=False,
        max_subqueries=-1,
    )
    plan = DefaultQueryPlanner(settings).plan(
        QueryRequest(text="查询", top_k=-4),
        make_analysis(),
        RewriteDecision(use_llm=True),
        make_rewrite(subqueries=("子查询",)),
    )

    assert plan.route_first is False
    assert plan.hybrid is False
    assert plan.rrf_k >= 1
    assert plan.max_attempts == 1
    assert plan.queries == ("主查询",)


def test_planner_rejects_non_boolean_feature_switches_and_invalid_hybrid_profiles() -> None:
    with pytest.raises(ValueError, match="route_enabled"):
        QueryPlannerSettings(route_enabled=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="hybrid_enabled"):
        QueryPlannerSettings(hybrid_enabled="yes")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="hybrid_profiles"):
        QueryPlannerSettings(hybrid_profiles=("aiops",))  # type: ignore[arg-type]


def test_planner_clamps_settings_and_request_top_k_to_configured_maximum() -> None:
    settings = QueryPlannerSettings(top_k=10_000, aiops_top_k=10_000, max_top_k=4)
    planner = DefaultQueryPlanner(settings)

    defaulted = planner.plan(
        QueryRequest(text="查询", profile="chat", top_k=False),
        make_analysis(profile=RetrievalProfile.CHAT),
        RewriteDecision(use_llm=False),
        make_rewrite(),
    )
    requested = planner.plan(
        QueryRequest(text="查询", profile="aiops", top_k=10_000),
        make_analysis(profile=RetrievalProfile.AIOPS),
        RewriteDecision(use_llm=False),
        make_rewrite(),
    )

    assert (defaulted.final_top_k, defaulted.candidate_k) == (4, 8)
    assert (requested.final_top_k, requested.candidate_k) == (4, 12)
    assert QueryPlannerSettings(top_k=True, aiops_top_k=0, max_top_k=4).top_k == 3  # type: ignore[arg-type]
    assert QueryPlannerSettings(top_k=True, aiops_top_k=0, max_top_k=4).aiops_top_k == 1  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="max_top_k"):
        QueryPlannerSettings(max_top_k=True)  # type: ignore[arg-type]


def test_inferred_and_valid_canonical_candidates_are_merged_without_mutation() -> None:
    registry = MetadataFieldRegistry(
        (
            MetadataFieldSpec("manufacturer", aliases=("厂商",)),
            MetadataFieldSpec("model", strict=True),
        )
    )
    analysis = make_analysis(inferred_filter={"model": "SUN2000"})
    rewrite = make_rewrite(
        entity_candidates=(
            EntityCandidate("厂商", "huawei", "华为"),
            EntityCandidate("unknown", "x", "x"),
        )
    )
    plan = DefaultQueryPlanner(metadata_registry=registry).plan(
        QueryRequest(text="华为逆变器"),
        analysis,
        RewriteDecision(use_llm=False),
        rewrite,
    )

    assert plan.metadata_filter == {"model": "SUN2000", "manufacturer": "huawei"}
    assert analysis.inferred_filter == {"model": "SUN2000"}
    assert rewrite.entity_candidates[0].field == "厂商"


def test_inferred_filter_wins_over_conflicting_candidates_and_strict_conflicts_fail_closed() -> (
    None
):
    non_strict = DefaultQueryPlanner().plan(
        QueryRequest(text="华为 INVT"),
        make_analysis(inferred_filter={"manufacturer": "huawei"}),
        RewriteDecision(use_llm=False),
        make_rewrite(entity_candidates=(EntityCandidate("manufacturer", "invt", "INVT"),)),
    )
    assert non_strict.metadata_filter == {"manufacturer": "huawei"}

    with pytest.raises(ValueError, match="model"):
        DefaultQueryPlanner().plan(
            QueryRequest(text="SUN2000 SUN3000"),
            make_analysis(inferred_filter={"model": "SUN2000"}),
            RewriteDecision(use_llm=False),
            make_rewrite(entity_candidates=(EntityCandidate("model", "SUN3000", "SUN3000"),)),
        )


def test_explicit_filter_ignores_candidates_and_inferred_values() -> None:
    plan = DefaultQueryPlanner().plan(
        QueryRequest(text="华为", metadata_filter={"model": "SUN2000"}),
        make_analysis(inferred_filter={"manufacturer": "huawei"}),
        RewriteDecision(use_llm=False),
        make_rewrite(entity_candidates=(EntityCandidate("manufacturer", "huawei", "华为"),)),
    )

    assert plan.metadata_filter == {"model": "SUN2000"}
    assert plan.explicit_filter is True


def test_explicit_empty_filter_remains_a_planner_boundary() -> None:
    request = QueryRequest.from_search_query(SearchQuery(text="查询", metadata_filter={}))
    plan = DefaultQueryPlanner().plan(
        request,
        make_analysis(inferred_filter={"manufacturer": "huawei"}),
        RewriteDecision(use_llm=False),
        make_rewrite(),
    )

    assert plan.metadata_filter == {}
    assert plan.explicit_filter is True
    assert plan.route_first is False


def test_candidate_validation_uses_request_profile_as_safety_boundary() -> None:
    registry = MetadataFieldRegistry(
        (MetadataFieldSpec("aiops_only", profiles=(RetrievalProfile.AIOPS,)),)
    )
    with pytest.raises(ValueError, match="profile"):
        DefaultQueryPlanner(metadata_registry=registry).plan(
            QueryRequest(text="value", profile=RetrievalProfile.CHAT),
            make_analysis(profile=RetrievalProfile.AIOPS, inferred_filter={}),
            RewriteDecision(use_llm=False),
            make_rewrite(entity_candidates=(EntityCandidate("aiops_only", "value", "value"),)),
        )


def test_planner_rejects_request_and_analysis_profile_mismatch() -> None:
    with pytest.raises(ValueError, match="profile"):
        DefaultQueryPlanner().plan(
            QueryRequest(text="查询", profile=RetrievalProfile.CHAT),
            make_analysis(profile=RetrievalProfile.AIOPS),
            RewriteDecision(use_llm=False),
            make_rewrite(),
        )


def test_planner_fails_closed_for_forged_candidate_evidence() -> None:
    plan = DefaultQueryPlanner().plan(
        QueryRequest(text="华为 SUN2000"),
        make_analysis(inferred_filter={}),
        RewriteDecision(use_llm=False),
        make_rewrite(entity_candidates=(EntityCandidate("model", "SUN2000", "华为"),)),
    )

    assert plan.metadata_filter == {}


def test_plan_preserves_all_upstream_stage_values() -> None:
    analysis = make_analysis(profile=RetrievalProfile.PROTOCOL)
    decision = RewriteDecision(use_llm=True)
    rewrite = make_rewrite(source="llm")

    plan = DefaultQueryPlanner().plan(
        QueryRequest(text="查询", profile=RetrievalProfile.PROTOCOL), analysis, decision, rewrite
    )

    assert plan.analysis.profile is RetrievalProfile.PROTOCOL
    assert plan.rewrite.source == "llm"
    assert plan.decision.use_llm is True


def test_plan_deeply_isolates_all_upstream_stage_values() -> None:
    request = QueryRequest(text="查询", metadata_filter={"request": {"items": ["a"]}})
    analysis = make_analysis(inferred_filter={"analysis": {"items": ["b"]}})
    decision = RewriteDecision(use_llm=False)
    rewrite = make_rewrite()

    plan = DefaultQueryPlanner().plan(request, analysis, decision, rewrite)
    request.metadata_filter["request"]["items"].append("changed")  # type: ignore[index]
    analysis.inferred_filter["analysis"]["items"].append("changed")

    assert plan.request.metadata_filter == {"request": {"items": ["a"]}}
    assert plan.analysis.inferred_filter == {"analysis": {"items": ["b"]}}
    assert plan.metadata_filter == {"request": {"items": ["a"]}}
    assert plan.request is not request
    assert plan.analysis is not analysis
    assert plan.decision is not decision
    assert plan.rewrite is not rewrite


def test_planner_rejects_an_empty_final_query_set() -> None:
    with pytest.raises(ValueError, match="query"):
        DefaultQueryPlanner().plan(
            QueryRequest(text="查询"),
            make_analysis(),
            RewriteDecision(use_llm=False),
            make_rewrite(primary_text="", subqueries=("",)),
        )


def test_plan_works_with_real_aiops_analysis_example() -> None:
    request = QueryRequest(text="华为逆变器告警怎么排查、处理并验证？", profile="aiops")
    analysis = DefaultQueryAnalyzer().analyze(request)
    plan = DefaultQueryPlanner(QueryPlannerSettings(max_subqueries=1)).plan(
        request,
        analysis,
        RewriteDecision(use_llm=False),
        make_rewrite(
            primary_text=analysis.normalized_text, subqueries=("华为逆变器告警排查", "超量")
        ),
    )

    assert plan.queries == (analysis.normalized_text, "华为逆变器告警排查")
    assert plan.metadata_filter is not None
    assert plan.metadata_filter["document_type"] == "runbook"
    assert plan.metadata_filter["manufacturer"] == "huawei"
