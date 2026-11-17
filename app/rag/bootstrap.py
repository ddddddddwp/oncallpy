"""Lazy composition root for default RAG dependencies."""

# 组合根负责把抽象端口绑定到默认基础设施；用例层本身不需要知道这些具体实现。
from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache
from typing import TYPE_CHECKING, Any, cast

from app.config import config
from app.rag.ports import SparseSearcher, VectorSearcher, VectorWriter

if TYPE_CHECKING:
    from app.rag.indexing.indexing_service import IndexingService
    from app.rag.infrastructure.dashscope_embeddings import DashScopeEmbeddings
    from app.rag.infrastructure.milvus_vector_store import MilvusVectorStore
    from app.rag.infrastructure.sqlite_sparse_index import SqliteSparseIndex
    from app.rag.query.contracts import QueryRewriteModel
    from app.rag.query.engine import QueryEngine
    from app.rag.query.query_service import QueryService


class _LazyPort:
    """Resolve an external adapter only when a use case invokes the port."""

    def __init__(self, provider: Callable[[], Any]) -> None:
        self.provider = provider

    def __getattr__(self, name: str) -> Any:
        # 第一次真正调用端口方法时才创建适配器，避免“仅导入模块”就连接外部服务。
        return getattr(self.provider(), name)


@lru_cache(maxsize=1)
def get_embedding_provider() -> DashScopeEmbeddings:
    """Return the shared DashScope embedding provider."""
    from app.rag.infrastructure.dashscope_embeddings import DashScopeEmbeddings

    # lru_cache 使整个进程复用同一个 provider，也避免重复构造客户端配置。
    return DashScopeEmbeddings(
        api_key=config.dashscope_api_key,
        model=config.dashscope_embedding_model,
        dimensions=1024,
    )


@lru_cache(maxsize=1)
def get_vector_store() -> MilvusVectorStore:
    """Return the shared compatibility-aware Milvus adapter."""
    # 保留兼容类作为默认入口，使旧调用方的 monkeypatch/构造约定继续生效。
    from app.rag.vector_store_manager import VectorStoreManager

    return VectorStoreManager()


@lru_cache(maxsize=1)
def get_sparse_index() -> SqliteSparseIndex:
    """Return the shared compatibility-aware SQLite sparse adapter."""
    from app.rag.sparse_index_service import SparseIndexService

    # SQLite 路径允许旧配置缺少新字段时回退到稳定默认值。
    return SparseIndexService(
        db_path=getattr(config, "rag_sparse_index_path", "data/rag_sparse.sqlite")
    )


@lru_cache(maxsize=1)
def get_indexing_service() -> IndexingService:
    """Return the shared indexing use case with explicit port dependencies."""
    from app.rag.indexing.deduplicator import RagDeduplicator
    from app.rag.indexing.document_loader import RagDocumentLoader
    from app.rag.indexing.document_splitter import DocumentSplitterService
    from app.rag.indexing.indexing_service import IndexingService
    from app.rag.indexing.metadata_builder import RagMetadataBuilder

    # 去重器和索引用例共享同一个惰性 VectorWriter，确保 generation 判断与写入指向同一存储。
    vector_writer = cast(VectorWriter, _LazyPort(get_vector_store))
    return IndexingService(
        loader=RagDocumentLoader(),
        splitter=DocumentSplitterService(),
        metadata_builder=RagMetadataBuilder(),
        deduplicator=RagDeduplicator(vector_writer),
        vector_writer=vector_writer,
        # sparse 使用 factory：只有检查/同步 FTS generation 时才打开 SQLite。
        sparse_writer_factory=get_sparse_index,
    )


@lru_cache(maxsize=1)
def get_query_rewrite_model() -> QueryRewriteModel:
    """Build the optional structured rewrite adapter only on first LLM use."""
    from app.core.llm_factory import LLMFactory
    from app.rag.infrastructure.langchain_query_rewrite_model import LangChainQueryRewriteModel

    return LangChainQueryRewriteModel(
        LLMFactory.create_chat_model(
            model=config.rag_query_llm_model or None,
            temperature=0.0,
            streaming=False,
            timeout=config.rag_query_llm_timeout_seconds,
        )
    )


def _query_planner_settings():
    """Map application settings to the bounded, framework-neutral planner."""
    from app.rag.models import RetrievalProfile
    from app.rag.query.planner import QueryPlannerSettings

    profiles: list[RetrievalProfile] = []
    for item in str(config.rag_hybrid_profiles).split(","):
        try:
            profiles.append(RetrievalProfile(item.strip()))
        except ValueError:
            continue
    return QueryPlannerSettings(
        top_k=config.rag_top_k,
        aiops_top_k=config.rag_aiops_top_k,
        route_enabled=config.rag_route_retrieval_enabled,
        hybrid_enabled=config.rag_hybrid_enabled,
        hybrid_profiles=tuple(profiles),
        rrf_k=config.rag_rrf_k,
        route_top_k=config.rag_route_top_k,
        sparse_top_k=config.rag_sparse_top_k,
        max_subqueries=config.rag_query_max_subqueries,
        max_attempts=config.rag_query_max_retrieval_attempts,
    )


@lru_cache(maxsize=1)
def get_query_engine() -> QueryEngine:
    """Compose the independent query pipeline without resolving external ports."""
    from app.rag.query.analyzer import DefaultQueryAnalyzer
    from app.rag.query.debugger import NoOpQueryDebugger
    from app.rag.query.engine import QueryEngine
    from app.rag.query.executor import RetrievalExecutor
    from app.rag.query.metadata_registry import default_metadata_registry
    from app.rag.query.planner import DefaultQueryPlanner
    from app.rag.query.query_router import DefaultQueryRouter
    from app.rag.query.reranker import LocalReranker
    from app.rag.query.retrieval_fusion import ReciprocalRankFusion, WeightedQueryFusion
    from app.rag.query.rewrite_policy import DefaultRewritePolicy, RewritePolicySettings
    from app.rag.query.rewriters.llm_rewriter import LLMQueryRewriter
    from app.rag.query.rewriters.rule_rewriter import RuleQueryRewriter

    registry = default_metadata_registry()
    analyzer = DefaultQueryAnalyzer(registry)
    router = DefaultQueryRouter(analyzer=analyzer, registry=registry)
    llm_model = cast("QueryRewriteModel", _LazyPort(get_query_rewrite_model))
    return QueryEngine(
        analyzer=analyzer,
        rewrite_policy=DefaultRewritePolicy(
            RewritePolicySettings(
                llm_enabled=config.rag_query_llm_enabled,
                confidence_threshold=config.rag_query_rewrite_confidence_threshold,
            )
        ),
        rule_rewriter=RuleQueryRewriter(),
        llm_rewriter=LLMQueryRewriter(
            llm_model,
            max_subqueries=config.rag_query_max_subqueries,
            registry=registry,
        ),
        planner=DefaultQueryPlanner(_query_planner_settings(), metadata_registry=registry),
        executor=RetrievalExecutor(
            vector_searcher=cast(VectorSearcher, _LazyPort(get_vector_store)),
            sparse_searcher=cast(SparseSearcher, _LazyPort(get_sparse_index)),
            router=router,
            fusion=ReciprocalRankFusion(),
            multi_query_fusion=WeightedQueryFusion(),
        ),
        reranker=LocalReranker(),
        debugger=NoOpQueryDebugger(),
    )


@lru_cache(maxsize=1)
def get_query_service() -> QueryService:
    """Return the legacy facade backed by the cached independent query engine."""
    from app.rag.query.query_service import QueryService

    # 查询用例依赖 Protocol；这里一次性装配路由、双路召回、融合与本地重排策略。
    return QueryService(
        engine=get_query_engine(),
        rewrite_mode=config.rag_query_rewrite_mode,
    )
