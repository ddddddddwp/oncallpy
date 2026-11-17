"""Reciprocal-rank fusion for framework-neutral search hits."""

# RRF 只使用名次而非原始分数，适合合并尺度完全不同的向量距离与 BM25 分数。
from __future__ import annotations

import hashlib
import math
import sys
from collections import defaultdict
from copy import deepcopy
from numbers import Real

from app.rag.models.document import Document
from app.rag.models.search_hit import SearchHit

MAX_RRF_K = 1_000_000


class ReciprocalRankFusion:
    """Merge dense and sparse ranks using the existing RRF formula."""

    def fuse(
        self,
        dense_results: list[SearchHit],
        sparse_results: list[SearchHit],
        k: int = 60,
    ) -> list[SearchHit]:
        # 三张表分别累计融合分、保留代表文档、记录该文档被哪些通道召回。
        scores: dict[str, float] = defaultdict(float)
        documents: dict[str, Document] = {}
        sources: dict[str, list[str]] = defaultdict(list)

        # rank 从 1 开始；越靠前的候选贡献 1/(k+rank) 越大。
        for source, results in (("dense", dense_results), ("sparse", sparse_results)):
            for rank, hit in enumerate(results, start=1):
                key = _document_key(hit.document)
                # 同一 chunk 两路命中时采用 sparse 文档对象，以保留其可能更完整的本地 metadata。
                if key not in documents or source == "sparse":
                    documents[key] = hit.document
                scores[key] += 1.0 / (k + rank)
                if source not in sources[key]:
                    sources[key].append(source)

        # 所有通道贡献相加后按 relevance 降序排列，原始距离不再直接参与比较。
        fused = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        results: list[SearchHit] = []
        for key, score in fused:
            document = documents[key]
            document.metadata["_retrieval_sources"] = sources[key]
            # 标记分数语义，避免 reranker 把 RRF relevance 再按“距离”做倒数转换。
            document.metadata["_retrieval_score_kind"] = "relevance"
            results.append(SearchHit(document=document, score=score, source="fused"))
        return results


class WeightedQueryFusion:
    """Fuse the ranked candidate lists produced by the planned queries.

    Unlike the dense/sparse RRF stage, the input weight represents the
    importance of a query variant.  The primary query therefore receives
    a larger contribution than generated subqueries.

    ``k`` is bounded by ``MAX_RRF_K`` so pathological integer inputs cannot
    overflow conversion during score calculation.
    """

    def fuse_ranked(
        self,
        ranked_results: list[tuple[float, list[SearchHit]]],
        k: int = 60,
    ) -> list[SearchHit]:
        if type(k) is not int or not 1 <= k <= MAX_RRF_K:
            raise ValueError("k must be a non-boolean positive integer within the maximum")

        scores: dict[str, float] = defaultdict(float)
        documents: dict[str, Document] = {}
        sources: dict[str, list[str]] = defaultdict(list)

        for weight, hits in ranked_results:
            if not _valid_weight(weight):
                continue
            seen_in_query: set[str] = set()
            for rank, hit in enumerate(hits, start=1):
                key = _document_key(hit.document)
                if key in seen_in_query:
                    continue
                seen_in_query.add(key)
                if key not in documents:
                    # The retrieval layer owns its annotations.  Never mutate
                    # a document that belongs to a store adapter or another stage.
                    documents[key] = Document(
                        page_content=hit.document.page_content,
                        metadata=deepcopy(hit.document.metadata),
                        id=hit.document.id,
                    )
                scores[key] = _saturating_add(scores[key], float(weight) / (k + rank))
                for source in _source_labels(hit):
                    if source not in sources[key]:
                        sources[key].append(source)

        results: list[SearchHit] = []
        # Python's sort is stable, so equal scores retain the first-seen key order.
        for key, score in sorted(scores.items(), key=lambda item: item[1], reverse=True):
            document = documents[key]
            document.metadata["_retrieval_sources"] = list(sources[key])
            document.metadata["_retrieval_score_kind"] = "relevance"
            results.append(SearchHit(document=document, score=score, source="fused"))
        return results


def _document_key(document: Document) -> str:
    metadata = document.metadata or {}
    # 优先使用稳定 chunk 身份；兼容旧数据时才逐级回退到旧 id 或内容前缀。
    for key in ("_chunk_id", "id", "chunk_id"):
        value = metadata.get(key)
        if value:
            return str(value)
    if document.id:
        return str(document.id)
    identity = f"{metadata.get('_source', '')}\0{document.page_content}".encode()
    return f"content:{hashlib.sha256(identity).hexdigest()}"


def _valid_weight(value: object) -> bool:
    if not isinstance(value, Real) or isinstance(value, bool):
        return False
    try:
        numeric = float(value)
    except OverflowError:
        return False
    return math.isfinite(numeric) and numeric > 0


def _saturating_add(left: float, right: float) -> float:
    """Keep ranking deterministic when valid finite weights approach float limits."""
    total = left + right
    return total if math.isfinite(total) else sys.float_info.max


def _source_labels(hit: SearchHit) -> list[str]:
    """Keep provenance from an already-fused input without sharing its list."""
    annotated = hit.document.metadata.get("_retrieval_sources")
    if isinstance(annotated, list) and all(isinstance(item, str) for item in annotated):
        return list(annotated)
    return [hit.source]
