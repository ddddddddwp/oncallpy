from __future__ import annotations

from dataclasses import replace

from app.rag.models import Document, RetrievalProfile, SearchHit
from app.rag.query.executor import RetrievalExecutor
from app.rag.query.models import (
    QueryAnalysis,
    QueryPlan,
    QueryRequest,
    RewriteDecision,
    RewrittenQuery,
)
from app.rag.query.retrieval_fusion import ReciprocalRankFusion, WeightedQueryFusion


class FakeVector:
    def __init__(self, responses: list[object]) -> None:
        self.responses = iter(responses)
        self.calls: list[tuple[str, int, dict[str, object] | None]] = []

    def search(self, query: str, k: int, metadata_filter=None):  # type: ignore[no-untyped-def]
        self.calls.append((query, k, metadata_filter))
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return response

    def supports_metadata_filter(self) -> bool:
        return True


class FakeRouter:
    def route_summary_filter(self, query_filter):  # type: ignore[no-untyped-def]
        return {"document_type": "route_summary"}

    def filter_from_route_metadata(self, metadata, profile):  # type: ignore[no-untyped-def]
        return {"_source": metadata["_source"], "document_type": "document"}

    def metadata_filter_fallbacks(self, metadata_filter):  # type: ignore[no-untyped-def]
        return [dict(metadata_filter)]

    def requires_taxonomy_match(self, metadata_filter):  # type: ignore[no-untyped-def]
        return "model" in metadata_filter


def hit(name: str, **metadata: object) -> SearchHit:
    return SearchHit(Document(name, {"_chunk_id": name, **metadata}), 1.0, "dense")


def plan(
    *,
    route_first: bool = True,
    metadata_filter=None,
    explicit_filter: bool = False,
    max_attempts: int = 8,
) -> QueryPlan:
    request = QueryRequest(text="primary", profile=RetrievalProfile.CHAT)
    analysis = QueryAnalysis(normalized_text="primary", profile=RetrievalProfile.CHAT)
    rewrite = RewrittenQuery(primary_text="primary", subqueries=("secondary", "third"))
    return QueryPlan(
        request=request,
        analysis=analysis,
        decision=RewriteDecision(False),
        rewrite=rewrite,
        queries=("primary", "secondary", "third"),
        metadata_filter=metadata_filter,
        final_top_k=3,
        candidate_k=5,
        route_first=route_first,
        hybrid=False,
        rrf_k=60,
        max_attempts=max_attempts,
        explicit_filter=explicit_filter,
    )


def executor(vector: FakeVector) -> RetrievalExecutor:
    return RetrievalExecutor(
        vector_searcher=vector,
        sparse_searcher=None,
        router=FakeRouter(),
        fusion=ReciprocalRankFusion(),
        multi_query_fusion=WeightedQueryFusion(),
    )


def test_route_is_probed_once_then_all_queries_share_selected_content_filter() -> None:
    vector = FakeVector(
        [
            [hit("route", document_type="route_summary", _source="manual")],
            [hit("primary")],
            [hit("secondary")],
            [hit("third")],
        ]
    )

    result = executor(vector).execute(plan())

    assert [call[0] for call in vector.calls] == ["primary", "primary", "secondary", "third"]
    assert vector.calls[0][2] == {"document_type": "route_summary"}
    assert all(
        call[2] == {"_source": "manual", "document_type": "document"} for call in vector.calls[1:]
    )
    assert result.attempts_used == 4


def test_attempt_budget_counts_dense_only_and_stops_all_store_calls() -> None:
    vector = FakeVector([[], []])

    result = executor(vector).execute(plan(route_first=False, metadata_filter=None, max_attempts=2))

    assert len(vector.calls) == 2
    assert result.attempts_used == 2
    assert result.degradation_reasons == ["budget_exhausted"]


def test_dense_result_survives_sparse_failure() -> None:
    class Sparse:
        def search_hits(self, query, k, metadata_filter=None):  # type: ignore[no-untyped-def]
            raise RuntimeError("down")

    vector = FakeVector([[hit("primary")]])
    retrieval = RetrievalExecutor(
        vector_searcher=vector,
        sparse_searcher=Sparse(),
        router=FakeRouter(),
        fusion=ReciprocalRankFusion(),
        multi_query_fusion=WeightedQueryFusion(),
    )
    single = replace(
        plan(
            route_first=False,
            metadata_filter={"manufacturer": "huawei"},
            explicit_filter=True,
        ),
        queries=("primary",),
        hybrid=True,
    )

    result = retrieval.execute(single)

    assert [item.document.page_content for item in result.candidates] == ["primary"]
    assert result.degradation_reasons == ["sparse_search_failed"]


def test_filtered_type_error_never_retries_an_unfiltered_search() -> None:
    vector = FakeVector([TypeError("filter unsupported")])

    result = executor(vector).execute(
        plan(
            route_first=False,
            metadata_filter={"model": "X"},
            explicit_filter=True,
            max_attempts=1,
        )
    )

    assert len(vector.calls) == 1
    assert vector.calls[0][2] == {"model": "X"}
    assert result.candidates == []


def test_strict_filter_miss_never_falls_back_to_unfiltered_search() -> None:
    vector = FakeVector([[], []])

    result = executor(vector).execute(plan(route_first=True, metadata_filter={"model": "X"}))

    assert all(call[2] is not None for call in vector.calls)
    assert result.degradation_reasons == ["route_miss", "filtered_search_empty"]


def test_explicit_filter_is_not_relaxed_when_routing_is_disabled() -> None:
    vector = FakeVector([[]])

    executor(vector).execute(
        plan(
            route_first=False,
            metadata_filter={"manufacturer": "huawei"},
            explicit_filter=True,
        )
    )
    assert [call[2] for call in vector.calls] == [
        {"manufacturer": "huawei"},
        {"manufacturer": "huawei"},
        {"manufacturer": "huawei"},
    ]


def test_unsupported_metadata_filter_skips_route_probe_at_execution_time() -> None:
    vector = FakeVector([[hit("primary")]])
    vector.supports_metadata_filter = lambda: False  # type: ignore[method-assign]

    result = executor(vector).execute(plan(route_first=True, metadata_filter=None))

    assert all(call[2] is None for call in vector.calls)
    assert [item.document.page_content for item in result.candidates] == ["primary"]


def test_unsupported_metadata_filter_fails_closed_without_calling_vector() -> None:
    vector = FakeVector([[hit("out-of-scope")]])
    vector.supports_metadata_filter = lambda: False  # type: ignore[method-assign]

    result = executor(vector).execute(
        plan(
            route_first=False,
            metadata_filter={"tenant": "allowed"},
            explicit_filter=True,
        )
    )

    assert vector.calls == []
    assert result.candidates == []
    assert result.degradation_reasons == ["metadata_filter_unsupported"]


def test_total_dense_outage_has_error_but_sparse_recovery_clears_it() -> None:
    failed = executor(FakeVector([RuntimeError("token=abc storage down")])).execute(
        replace(plan(route_first=False, max_attempts=1), queries=("primary",))
    )
    assert failed.error == "token=*** storage down"

    class Sparse:
        def search_hits(self, query, k, metadata_filter=None):  # type: ignore[no-untyped-def]
            return [SearchHit(Document("sparse"), 1.0, "sparse")]

    recovered = RetrievalExecutor(
        vector_searcher=FakeVector([RuntimeError("down")]),
        sparse_searcher=Sparse(),
        router=FakeRouter(),
        fusion=ReciprocalRankFusion(),
        multi_query_fusion=WeightedQueryFusion(),
    ).execute(
        replace(
            plan(route_first=False, max_attempts=1),
            queries=("primary",),
            hybrid=True,
        )
    )
    assert recovered.candidates
    assert recovered.error is None


def test_dense_failure_does_not_prevent_later_planned_queries() -> None:
    vector = FakeVector([RuntimeError("temporary"), [hit("secondary")], []])

    result = executor(vector).execute(plan(route_first=False, metadata_filter=None))

    assert [item.document.page_content for item in result.candidates] == ["secondary"]
    assert result.degradation_reasons == ["dense_search_failed"]


def test_routing_disabled_keeps_inferred_filter_fallbacks_relaxable() -> None:
    class FallbackRouter(FakeRouter):
        def metadata_filter_fallbacks(self, metadata_filter):  # type: ignore[no-untyped-def]
            return [dict(metadata_filter), {}]

    vector = FakeVector([[], [hit("fallback")], [], []])
    retrieval = RetrievalExecutor(
        vector_searcher=vector,
        sparse_searcher=None,
        router=FallbackRouter(),
        fusion=ReciprocalRankFusion(),
        multi_query_fusion=WeightedQueryFusion(),
    )

    result = retrieval.execute(plan(route_first=False, metadata_filter={"manufacturer": "huawei"}))

    assert vector.calls[0][2] == {"manufacturer": "huawei"}
    assert vector.calls[1][2] == {}
    assert [item.document.page_content for item in result.candidates] == ["fallback"]


def test_route_metadata_and_filter_are_deeply_isolated_on_returned_hits() -> None:
    route = hit("route", document_type="route_summary", _source="manual", nested={"a": []})
    content = hit("content")
    vector = FakeVector([[route], [content], [], []])

    result = executor(vector).execute(plan())

    metadata = result.candidates[0].document.metadata
    assert metadata["_retrieval_filter"] == {"_source": "manual", "document_type": "document"}
    assert metadata["_retrieval_route"]["_source"] == "manual"
    metadata["_retrieval_route"]["nested"]["a"].append("changed")
    metadata["_retrieval_filter"]["_source"] = "changed"
    assert route.document.metadata["nested"] == {"a": []}
    assert content.document.metadata.get("_retrieval_filter") is None
    assert result.degraded is False


def test_sparse_hit_is_retained_after_dense_failure_without_extra_attempt() -> None:
    class Sparse:
        def search_hits(self, query, k, metadata_filter=None):  # type: ignore[no-untyped-def]
            return [SearchHit(Document("sparse", {"_chunk_id": "sparse"}), 1.0, "sparse")]

    vector = FakeVector([RuntimeError("dense down")])
    retrieval = RetrievalExecutor(
        vector_searcher=vector,
        sparse_searcher=Sparse(),
        router=FakeRouter(),
        fusion=ReciprocalRankFusion(),
        multi_query_fusion=WeightedQueryFusion(),
    )
    value = replace(
        plan(route_first=False, metadata_filter=None, max_attempts=1),
        queries=("primary",),
        hybrid=True,
    )

    result = retrieval.execute(value)

    assert [item.document.page_content for item in result.candidates] == ["sparse"]
    assert result.attempts_used == 1
    assert result.degradation_reasons == ["dense_search_failed"]
    assert result.degraded is True


def test_route_filters_are_deduplicated_across_ranked_route_hits() -> None:
    class RouteRouter(FakeRouter):
        def filter_from_route_metadata(self, metadata, profile):  # type: ignore[no-untyped-def]
            return {"_source": metadata["_source"]}

    routes = [
        hit("route-a", document_type="route_summary", _source="A"),
        hit("route-a-copy", document_type="route_summary", _source="A"),
        hit("route-b", document_type="route_summary", _source="B"),
    ]
    vector = FakeVector([routes, [], [hit("from-b")]])
    retrieval = RetrievalExecutor(
        vector_searcher=vector,
        sparse_searcher=None,
        router=RouteRouter(),
        fusion=ReciprocalRankFusion(),
        multi_query_fusion=WeightedQueryFusion(),
    )

    result = retrieval.execute(replace(plan(max_attempts=3), queries=("primary",)))

    assert [call[2] for call in vector.calls] == [
        {"document_type": "route_summary"},
        {"_source": "A"},
        {"_source": "B"},
    ]
    assert [item.document.page_content for item in result.candidates] == ["from-b"]


def test_retrieval_boundaries_and_fusion_failures_fall_back_without_crashing() -> None:
    class BrokenFusion:
        def fuse(self, dense_results, sparse_results, k):  # type: ignore[no-untyped-def]
            raise RuntimeError("fusion down")

    class BrokenMultiFusion:
        def fuse_ranked(self, ranked_results, k):  # type: ignore[no-untyped-def]
            raise RuntimeError("multi fusion down")

    class Sparse:
        def search_hits(self, query, k, metadata_filter=None):  # type: ignore[no-untyped-def]
            return [SearchHit(Document("sparse", {"_chunk_id": "sparse"}), 1.0, "sparse")]

    vector = FakeVector([[hit("dense")]])
    retrieval = RetrievalExecutor(
        vector_searcher=vector,
        sparse_searcher=Sparse(),
        router=FakeRouter(),
        fusion=BrokenFusion(),
        multi_query_fusion=BrokenMultiFusion(),
    )
    value = replace(plan(route_first=False, max_attempts=1), queries=("primary",), hybrid=True)

    result = retrieval.execute(value)

    assert [item.document.page_content for item in result.candidates] == ["dense", "sparse"]
    assert result.degraded is False


def test_none_and_uncopyable_dense_results_become_dense_degradation() -> None:
    vector = FakeVector([None])

    result = executor(vector).execute(
        replace(plan(route_first=False, max_attempts=1), queries=("primary",))
    )

    assert result.candidates == []
    assert result.degradation_reasons == ["dense_search_failed"]


def test_uncopyable_dense_metadata_and_broken_sparse_iterator_degrade_per_source() -> None:
    class Uncopyable:
        def __deepcopy__(self, memo):  # type: ignore[no-untyped-def]
            raise RuntimeError("cannot copy")

    vector = FakeVector([[SearchHit(Document("bad", {"value": Uncopyable()}), 1.0, "dense")]])
    dense_result = executor(vector).execute(
        replace(plan(route_first=False, max_attempts=1), queries=("primary",))
    )
    assert dense_result.degradation_reasons == ["dense_search_failed"]

    class Sparse:
        def search_hits(self, query, k, metadata_filter=None):  # type: ignore[no-untyped-def]
            yield SearchHit(Document("sparse", {"_chunk_id": "sparse"}), 1.0, "sparse")
            raise RuntimeError("iterator down")

    vector = FakeVector([[hit("dense")]])
    retrieval = RetrievalExecutor(
        vector_searcher=vector,
        sparse_searcher=Sparse(),
        router=FakeRouter(),
        fusion=ReciprocalRankFusion(),
        multi_query_fusion=WeightedQueryFusion(),
    )
    result = retrieval.execute(
        replace(plan(route_first=False, max_attempts=1), queries=("primary",), hybrid=True)
    )
    assert [item.document.page_content for item in result.candidates] == ["dense"]
    assert result.degradation_reasons == ["sparse_search_failed"]


def test_hybrid_uses_separate_route_and_sparse_retrieval_depths() -> None:
    class Sparse:
        def __init__(self) -> None:
            self.calls: list[tuple[int, dict[str, object] | None]] = []

        def search_hits(self, query, k, metadata_filter=None):  # type: ignore[no-untyped-def]
            self.calls.append((k, metadata_filter))
            return []

    sparse = Sparse()
    vector = FakeVector(
        [
            [hit("route", document_type="route_summary", _source="manual")],
            [hit("content")],
        ]
    )
    retrieval = RetrievalExecutor(
        vector_searcher=vector,
        sparse_searcher=sparse,
        router=FakeRouter(),
        fusion=ReciprocalRankFusion(),
        multi_query_fusion=WeightedQueryFusion(),
    )
    value = replace(
        plan(max_attempts=2),
        queries=("primary",),
        hybrid=True,
        route_k=2,
        candidate_k=5,
        sparse_min_k=20,
    )

    retrieval.execute(value)

    assert [call[1] for call in vector.calls] == [2, 5]
    assert [call[0] for call in sparse.calls] == [20, 20]


def test_hybrid_off_never_calls_sparse_even_when_a_sparse_port_exists() -> None:
    class Sparse:
        def __init__(self) -> None:
            self.called = False

        def search_hits(self, query, k, metadata_filter=None):  # type: ignore[no-untyped-def]
            self.called = True
            return []

    sparse = Sparse()
    vector = FakeVector([[hit("primary")]])
    retrieval = RetrievalExecutor(
        vector_searcher=vector,
        sparse_searcher=sparse,
        router=FakeRouter(),
        fusion=ReciprocalRankFusion(),
        multi_query_fusion=WeightedQueryFusion(),
    )

    retrieval.execute(
        replace(plan(route_first=False, max_attempts=1), queries=("primary",), hybrid=False)
    )

    assert sparse.called is False
