"""Backward-compatible public facade over :class:`QueryEngine`."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import Any, cast

from loguru import logger

from app.config import config
from app.core.observability import get_trace_context, trace_span
from app.rag.models import Document, RetrievalProfile, SearchQuery
from app.rag.ports.sparse_searcher import SparseSearcher
from app.rag.ports.vector_searcher import VectorSearcher
from app.rag.query.analyzer import DefaultQueryAnalyzer
from app.rag.query.contracts import QueryRouter, Reranker, RetrievalFusion
from app.rag.query.debugger import NoOpQueryDebugger
from app.rag.query.engine import QueryEngine
from app.rag.query.errors import safe_error_summary
from app.rag.query.executor import RetrievalExecutor
from app.rag.query.models import QueryRequest, RewriteMode
from app.rag.query.planner import DefaultQueryPlanner, QueryPlannerSettings
from app.rag.query.retrieval_fusion import WeightedQueryFusion
from app.rag.query.rewrite_policy import DefaultRewritePolicy, RewritePolicySettings
from app.rag.query.rewriters.rule_rewriter import RuleQueryRewriter

_UNSET = object()


class QueryService:
    """Legacy retrieval API backed by an independently testable QueryEngine."""

    def __init__(
        self,
        *,
        engine: QueryEngine | object = _UNSET,
        vector_searcher: VectorSearcher | object = _UNSET,
        sparse_searcher: SparseSearcher | None | object = _UNSET,
        router: QueryRouter | object = _UNSET,
        fusion: RetrievalFusion | object = _UNSET,
        reranker: Reranker | object = _UNSET,
        rewrite_mode: RewriteMode | str = RewriteMode.OFF,
        trace_span_factory: Callable[..., Any] = trace_span,
    ) -> None:
        legacy = (vector_searcher, sparse_searcher, router, fusion, reranker)
        if engine is not _UNSET:
            if any(value is not _UNSET for value in legacy):
                raise ValueError(
                    "engine mode cannot be combined with legacy retrieval dependencies"
                )
            if engine is None:
                raise ValueError("engine must not be None")
            self.engine: QueryEngine = cast(QueryEngine, engine)
        else:
            missing = [
                name
                for name, value in (
                    ("vector_searcher", vector_searcher),
                    ("router", router),
                    ("fusion", fusion),
                    ("reranker", reranker),
                )
                if value is _UNSET or value is None
            ]
            if missing:
                raise ValueError("legacy mode requires " + ", ".join(missing))
            sparse = None if sparse_searcher is _UNSET else sparse_searcher
            self.engine = self._legacy_engine(
                vector_searcher,
                sparse,
                router,
                fusion,
                reranker,  # type: ignore[arg-type]
            )
        self.rewrite_mode = rewrite_mode
        self.trace_span_factory = trace_span_factory

    def _legacy_engine(self, vector, sparse, router, fusion, reranker):  # type: ignore[no-untyped-def]
        settings = _planner_settings()
        return QueryEngine(
            DefaultQueryAnalyzer(),
            DefaultRewritePolicy(RewritePolicySettings(llm_enabled=False)),
            RuleQueryRewriter(),
            None,
            DefaultQueryPlanner(settings),
            RetrievalExecutor(
                vector_searcher=vector,
                sparse_searcher=sparse,
                router=router,
                fusion=fusion,
                multi_query_fusion=WeightedQueryFusion(),
            ),
            reranker,
            NoOpQueryDebugger(),
        )

    def retrieve(self, query: SearchQuery) -> list[Document]:
        selected_profile = _normalize_profile(query.profile)
        final_top_k = query.top_k or _default_top_k(selected_profile)
        request = replace(
            QueryRequest.from_search_query(query), rewrite_mode=self.rewrite_mode, debug=True
        )
        explicit_filter = dict(query.metadata_filter) if query.metadata_filter else None
        input_data = {
            **get_trace_context(),
            "query": query.text,
            "profile": selected_profile.value,
            "top_k": final_top_k,
        }
        if explicit_filter:
            input_data["metadata_filter"] = explicit_filter
        with self.trace_span_factory("rag_retrieve", input_data=input_data) as span:
            try:
                result = self.engine.query(request)
            except Exception as exc:
                message = safe_error_summary(exc)
                logger.error(f"RAG 检索失败: {message}")
                span.set_output({"status": "error", "message": message})
                return []
            if result.error is not None:
                message = safe_error_summary(result.error)
                span.set_output(
                    {
                        "status": "error",
                        "message": message,
                        "candidate_k": result.plan.candidate_k,
                    }
                )
                return []
            documents = result.documents
            metadata_filter = result.plan.metadata_filter
            if documents:
                metadata_filter = documents[0].metadata.get("_retrieval_filter", metadata_filter)
            # The legacy span schema represented the unfiltered case as null.
            if metadata_filter == {}:
                metadata_filter = None
            route_matched = bool(documents and documents[0].metadata.get("_retrieval_route"))
            span.set_output(
                {
                    "status": "success",
                    "documents": len(documents),
                    "candidate_k": result.plan.candidate_k,
                    "metadata_filter": metadata_filter,
                    "route_attempted": any(
                        event.stage == "route" and event.status == "started"
                        for event in result.trace
                    ),
                    "route_matched": route_matched,
                    "fallback_reason": result.degradation_reasons[-1]
                    if result.degradation_reasons
                    else None,
                }
            )
            return documents


def _planner_settings() -> QueryPlannerSettings:
    profiles: list[RetrievalProfile] = []
    for item in str(getattr(config, "rag_hybrid_profiles", "aiops,protocol")).split(","):
        try:
            profiles.append(RetrievalProfile(item.strip()))
        except ValueError:
            continue
    return QueryPlannerSettings(
        top_k=getattr(config, "rag_top_k", 3),
        aiops_top_k=getattr(config, "rag_aiops_top_k", 8),
        route_enabled=bool(getattr(config, "rag_route_retrieval_enabled", True)),
        hybrid_enabled=bool(getattr(config, "rag_hybrid_enabled", True)),
        hybrid_profiles=tuple(profiles),
        rrf_k=getattr(config, "rag_rrf_k", 60),
        route_top_k=getattr(config, "rag_route_top_k", 2),
        sparse_top_k=getattr(config, "rag_sparse_top_k", 20),
    )


def _normalize_profile(profile: RetrievalProfile | str) -> RetrievalProfile:
    try:
        return RetrievalProfile(profile)
    except ValueError:
        return RetrievalProfile.CHAT


def _default_top_k(profile: RetrievalProfile) -> int:
    if profile is RetrievalProfile.AIOPS:
        return int(getattr(config, "rag_aiops_top_k", 8))
    return int(getattr(config, "rag_top_k", 3))
