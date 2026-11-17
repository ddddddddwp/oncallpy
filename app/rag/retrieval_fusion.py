"""Compatibility adapter for dense/sparse reciprocal-rank fusion."""

# 该函数保留旧的 LangChain tuple API，排序算法本身委托给领域层 ReciprocalRankFusion。
from __future__ import annotations

from typing import Literal

from langchain_core.documents import Document as LangChainDocument

from app.rag.models.document import Document
from app.rag.models.search_hit import SearchHit
from app.rag.query.retrieval_fusion import ReciprocalRankFusion, _document_key


def reciprocal_rank_fusion(
    dense_results: list[tuple[LangChainDocument, float]],
    sparse_results: list[tuple[LangChainDocument, float]],
    k: int = 60,
) -> list[tuple[LangChainDocument, float]]:
    """Preserve the tuple API while delegating ranking to the domain component."""
    # originals 保存旧调用方传入的对象，融合完成后可以返回相同的 LangChain 实例。
    originals: dict[str, LangChainDocument] = {}

    def convert(
        source: Literal["dense", "sparse"],
        results: list[tuple[LangChainDocument, float]],
    ) -> list[SearchHit]:
        # 入口转换为领域 SearchHit，让核心算法不依赖 LangChain 类型。
        hits: list[SearchHit] = []
        for document, score in results:
            domain_document = Document(
                page_content=document.page_content,
                metadata=document.metadata,
                id=getattr(document, "id", None),
            )
            key = _document_key(domain_document)
            # 与领域融合一致，同键时优先保留 sparse 通道对象作为最终兼容返回值。
            if key not in originals or source == "sparse":
                originals[key] = document
            hits.append(SearchHit(domain_document, float(score), source))
        return hits

    # 分数已经由 RRF 重算，返回 tuple 的第二项不再是原始距离/BM25 分数。
    fused = ReciprocalRankFusion().fuse(
        dense_results=convert("dense", dense_results),
        sparse_results=convert("sparse", sparse_results),
        k=k,
    )
    return [(originals[_document_key(hit.document)], hit.score) for hit in fused]


__all__ = ["reciprocal_rank_fusion"]
