"""Framework-neutral public API for the modular RAG layer."""

# 这个包入口只暴露领域模型与用例服务，避免调用方依赖具体向量库或 LangChain 类型。
from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    # 静态检查器需要看到真实类型；运行时则不导入这些重模块，以免触发外部依赖初始化。
    from app.rag.indexing.indexing_service import IndexingService
    from app.rag.models.document import Document
    from app.rag.models.indexing_result import IndexingResult
    from app.rag.models.search_hit import SearchHit
    from app.rag.models.search_query import SearchQuery
    from app.rag.query.query_service import QueryService

__all__ = [
    "Document",
    "IndexingResult",
    "IndexingService",
    "QueryService",
    "SearchHit",
    "SearchQuery",
]

# 名称到“模块 + 属性”的映射让公开 API 保持稳定，同时把真实导入推迟到首次访问。
_LAZY_EXPORTS = {
    "Document": ("app.rag.models.document", "Document"),
    "IndexingResult": ("app.rag.models.indexing_result", "IndexingResult"),
    "IndexingService": ("app.rag.indexing.indexing_service", "IndexingService"),
    "QueryService": ("app.rag.query.query_service", "QueryService"),
    "SearchHit": ("app.rag.models.search_hit", "SearchHit"),
    "SearchQuery": ("app.rag.models.search_query", "SearchQuery"),
}


def __getattr__(name: str) -> Any:
    # PEP 562 会在普通属性查找失败时调用这里，因此未知名称仍要表现为标准 AttributeError。
    try:
        module_name, attribute_name = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

    # 成功解析后写回模块全局命名空间；后续访问直接命中缓存，不再重复动态导入。
    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value
