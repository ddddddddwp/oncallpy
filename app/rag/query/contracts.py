"""Structural contracts used inside the query module."""

# 查询内部继续依赖 Protocol，路由、融合和重排策略可以独立替换或在测试中注入。
from __future__ import annotations

from typing import Any, Protocol

from app.rag.models.document import Document
from app.rag.models.search_hit import SearchHit
from app.rag.models.search_query import RetrievalProfile, SearchQuery
from app.rag.query.models import (
    QueryAnalysis,
    QueryExecution,
    QueryPlan,
    QueryRequest,
    QueryTraceEvent,
    RewriteDecision,
    RewrittenQuery,
)


class QueryRouter(Protocol):
    # 从自然语言与 profile 推断第一组可下推 metadata filter。
    def infer_metadata_filter(
        self,
        query: str,
        profile: RetrievalProfile,
    ) -> dict[str, Any]: ...

    # 明确定义允许放宽的过滤序列，QueryService 不自行猜测哪些条件可丢弃。
    def metadata_filter_fallbacks(
        self,
        metadata_filter: dict[str, Any],
    ) -> list[dict[str, Any]]: ...

    # route-first 第一阶段只搜索轻量摘要，第二阶段再从摘要生成内容过滤器。
    def route_summary_filter(
        self,
        query_filter: dict[str, Any] | None,
    ) -> dict[str, Any]: ...

    def filter_from_route_metadata(
        self,
        metadata: dict[str, Any],
        profile: RetrievalProfile,
    ) -> dict[str, Any]: ...

    def requires_taxonomy_match(self, metadata_filter: dict[str, Any]) -> bool: ...


class RetrievalFusion(Protocol):
    # 融合接收领域 SearchHit，因此算法不依赖 Milvus 距离或 SQLite row 类型。
    def fuse(
        self,
        dense_results: list[SearchHit],
        sparse_results: list[SearchHit],
        k: int,
    ) -> list[SearchHit]: ...


class Reranker(Protocol):
    # reranker 在融合后统一解释分数，并结合 query/profile 产生最终顺序。
    def rerank(
        self,
        query: str,
        candidates: list[SearchHit],
        profile: RetrievalProfile,
    ) -> list[SearchHit]: ...


class QueryUseCase(Protocol):
    # 对外只返回领域 Document；Agent/LangChain 转换由更外层边界处理。
    def retrieve(self, query: SearchQuery) -> list[Document]: ...


class QueryAnalyzer(Protocol):
    def analyze(self, request: QueryRequest) -> QueryAnalysis: ...


class RewritePolicy(Protocol):
    def decide(
        self,
        request: QueryRequest,
        analysis: QueryAnalysis,
    ) -> RewriteDecision: ...


class QueryRewriter(Protocol):
    def rewrite(
        self,
        request: QueryRequest,
        analysis: QueryAnalysis,
    ) -> RewrittenQuery: ...


class QueryRewriteModel(Protocol):
    def generate(self, payload: dict[str, Any]) -> dict[str, Any]: ...


class QueryPlanner(Protocol):
    def plan(
        self,
        request: QueryRequest,
        analysis: QueryAnalysis,
        decision: RewriteDecision,
        rewrite: RewrittenQuery,
    ) -> QueryPlan: ...


class QueryExecutor(Protocol):
    def execute(
        self,
        plan: QueryPlan,
        debugger: QueryDebugger | None = None,
    ) -> QueryExecution: ...


class QueryDebugger(Protocol):
    def emit(self, event: QueryTraceEvent) -> None: ...


class MultiQueryFusion(Protocol):
    def fuse_ranked(
        self,
        ranked_results: list[tuple[float, list[SearchHit]]],
        k: int,
    ) -> list[SearchHit]: ...
