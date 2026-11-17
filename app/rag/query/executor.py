"""Bounded execution of declarative query plans against retrieval ports."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from app.rag.models import Document, SearchHit
from app.rag.ports.sparse_searcher import SparseSearcher
from app.rag.ports.vector_searcher import VectorSearcher
from app.rag.query.contracts import (
    MultiQueryFusion,
    QueryDebugger,
    QueryRouter,
    RetrievalFusion,
)
from app.rag.query.errors import safe_error_summary
from app.rag.query.models import QueryExecution, QueryPlan, QueryTraceEvent
from app.rag.query.retrieval_fusion import _document_key


class RetrievalExecutor:
    """Execute one plan without widening caller-approved metadata filters.

    Dense calls are the only logical attempts: a paired sparse lookup is a
    degradation-tolerant companion, not another chance to hit the store.
    """

    def __init__(
        self,
        *,
        vector_searcher: VectorSearcher,
        sparse_searcher: SparseSearcher | None,
        router: QueryRouter,
        fusion: RetrievalFusion,
        multi_query_fusion: MultiQueryFusion,
    ) -> None:
        self._vector_searcher = vector_searcher
        self._sparse_searcher = sparse_searcher
        self._router = router
        self._fusion = fusion
        self._multi_query_fusion = multi_query_fusion

    def execute(
        self,
        plan: QueryPlan,
        debugger: QueryDebugger | None = None,
    ) -> QueryExecution:
        reasons: list[str] = []
        attempts_used = 0
        dense_error: str | None = None
        limit = _attempt_limit(plan.max_attempts)

        def degrade(reason: str) -> None:
            if reason not in reasons:
                reasons.append(reason)

        def dense(
            query: str,
            metadata_filter: dict[str, Any] | None,
            k: int,
        ) -> list[SearchHit] | None:
            nonlocal attempts_used, dense_error
            if attempts_used >= limit:
                degrade("budget_exhausted")
                return None
            attempts_used += 1
            try:
                # Supplying a filter is intentional.  A TypeError must not cause
                # a silent, unsafe retry without it.
                hits = self._vector_searcher.search(
                    query,
                    k=k,
                    metadata_filter=deepcopy(metadata_filter),
                )
                return _copy_hits(hits)
            except Exception as exc:
                if dense_error is None:
                    dense_error = safe_error_summary(exc)
                degrade("dense_search_failed")
                return []

        def retrieve(
            query: str,
            metadata_filter: dict[str, Any] | None,
            dense_k: int = plan.candidate_k,
        ) -> list[SearchHit] | None:
            dense_hits = dense(query, metadata_filter, dense_k)
            if dense_hits is None:
                return None
            if not (plan.hybrid and self._sparse_searcher is not None):
                return dense_hits
            try:
                sparse_hits = self._sparse_searcher.search_hits(
                    query,
                    k=max(plan.sparse_min_k, dense_k),
                    metadata_filter=deepcopy(metadata_filter),
                )
                sparse_hits = _copy_hits(sparse_hits)
            except Exception:
                degrade("sparse_search_failed")
                return dense_hits
            if not sparse_hits:
                return dense_hits
            try:
                return self._fusion.fuse(dense_hits, sparse_hits, plan.rrf_k)
            except Exception:
                return _stable_unique_hits(dense_hits, sparse_hits)

        primary = plan.queries[0]
        selected_filter: dict[str, Any] | None = None
        primary_hits: list[SearchHit] | None = None
        route_metadata: dict[str, Any] | None = None
        attempted_route_filters: list[dict[str, Any]] = []
        route_attempted = False
        route_matched = False

        # Capability checks belong to execution: adapters may lazily resolve a
        # store, while planning must remain a pure in-memory transformation.
        if plan.route_first and self._supports_metadata_filter():
            route_attempted = True
            self._emit(debugger, "route", "started")
            route_hits = retrieve(
                primary,
                self._router.route_summary_filter(plan.metadata_filter),
                plan.route_k,
            )
            if route_hits:
                route_hits = [
                    hit
                    for hit in route_hits
                    if hit.document.metadata.get("document_type") == "route_summary"
                ]
            for route_hit in route_hits or []:
                route_filter = self._router.filter_from_route_metadata(
                    route_hit.document.metadata, plan.analysis.profile
                )
                if not route_filter:
                    continue
                for candidate_filter in _unique_filters(
                    self._router.metadata_filter_fallbacks(route_filter)
                ):
                    if candidate_filter in attempted_route_filters:
                        continue
                    attempted_route_filters.append(deepcopy(candidate_filter))
                    attempt = retrieve(primary, candidate_filter)
                    if attempt is None:
                        break
                    if attempt:
                        selected_filter = candidate_filter
                        primary_hits = attempt
                        route_metadata = deepcopy(route_hit.document.metadata)
                        route_matched = True
                        break
                if primary_hits is not None or attempts_used >= limit:
                    break
            if primary_hits is None:
                degrade("route_miss")
                self._emit(debugger, "route", "degraded")
            else:
                self._emit(debugger, "route", "completed")

        if primary_hits is None:
            selected_filter, primary_hits = self._select_non_route_filter(
                plan, primary, retrieve, degrade
            )

        per_query: list[tuple[float, list[SearchHit]]] = []
        # For unfiltered retrieval a primary miss does not invalidate useful
        # subqueries; for a selected filter, it was already established above.
        if primary_hits is not None:
            _annotate_hits(primary_hits, selected_filter, route_metadata)
            per_query.append((1.0, primary_hits))
            for query in plan.queries[1:]:
                hits = retrieve(query, selected_filter)
                if hits is None:
                    break
                _annotate_hits(hits, selected_filter, route_metadata)
                per_query.append((0.6, hits))

        if per_query:
            try:
                candidates = self._multi_query_fusion.fuse_ranked(per_query, plan.rrf_k)
            except Exception:
                candidates = _stable_unique_hits(*(hits for _weight, hits in per_query))
        else:
            candidates = []
        execution = QueryExecution(
            candidates=candidates,
            degraded=bool(reasons),
            degradation_reasons=reasons,
            attempts_used=attempts_used,
            error=dense_error if not candidates else None,
            route_attempted=route_attempted,
            route_matched=route_matched,
            metadata_filter=deepcopy(selected_filter),
        )
        self._emit(debugger, "retrieve", "degraded" if execution.degraded else "completed")
        return execution

    def _supports_metadata_filter(self) -> bool:
        try:
            return bool(self._vector_searcher.supports_metadata_filter())
        except Exception:
            return False

    def _select_non_route_filter(self, plan, primary, retrieve, degrade):  # type: ignore[no-untyped-def]
        """Select a legal shared filter, preserving explicit-filter boundaries."""
        metadata_filter = plan.metadata_filter
        if (
            metadata_filter is not None
            and (bool(metadata_filter) or plan.explicit_filter)
            and not self._supports_metadata_filter()
        ):
            degrade("metadata_filter_unsupported")
            return metadata_filter, None
        # The explicit-filter marker is a hard caller boundary, including an
        # explicitly empty filter.  Routing may be disabled for inferred filters.
        if plan.explicit_filter:
            return metadata_filter, retrieve(primary, metadata_filter)

        if not metadata_filter:
            return None, retrieve(primary, None)

        for candidate_filter in _unique_filters(
            self._router.metadata_filter_fallbacks(metadata_filter)
        ):
            hits = retrieve(primary, candidate_filter)
            if hits is None or hits:
                return candidate_filter, hits

        degrade("filtered_search_empty")
        if self._router.requires_taxonomy_match(metadata_filter):
            # A strict taxonomy miss is fail-closed: there is no selected
            # content scope and subqueries must not probe another one.
            return metadata_filter, None
        return None, retrieve(primary, None)

    @staticmethod
    def _emit(debugger: QueryDebugger | None, stage: str, status: str) -> None:
        if debugger is None:
            return
        debugger.emit(QueryTraceEvent(stage=stage, status=status))  # type: ignore[arg-type]


def _attempt_limit(value: object) -> int:
    return value if type(value) is int and value > 0 else 1


def _unique_filters(filters: list[dict[str, Any]]) -> list[dict[str, Any]]:
    unique: list[dict[str, Any]] = []
    for value in filters:
        copied = deepcopy(value)
        if copied not in unique:
            unique.append(copied)
    return unique


def _copy_hits(hits: list[SearchHit]) -> list[SearchHit]:
    return [
        SearchHit(
            document=Document(
                page_content=hit.document.page_content,
                metadata=deepcopy(hit.document.metadata),
                id=hit.document.id,
            ),
            score=hit.score,
            source=hit.source,
        )
        for hit in hits
    ]


def _stable_unique_hits(*groups: list[SearchHit]) -> list[SearchHit]:
    """Fallback when a fusion implementation fails after retrieval succeeded."""
    unique: list[SearchHit] = []
    seen: set[str] = set()
    for hits in groups:
        for hit in hits:
            key = _document_key(hit.document)
            if key not in seen:
                seen.add(key)
                unique.append(hit)
    return unique


def _annotate_hits(
    hits: list[SearchHit],
    metadata_filter: dict[str, Any] | None,
    route_metadata: dict[str, Any] | None,
) -> None:
    """Attach executor-owned trace metadata without leaking mutable inputs."""
    for hit in hits:
        hit.document.metadata["_retrieval_filter"] = deepcopy(metadata_filter or {})
        if route_metadata is not None:
            hit.document.metadata["_retrieval_route"] = deepcopy(route_metadata)
