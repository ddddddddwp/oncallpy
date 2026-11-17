"""Local deterministic reranking for retrieved RAG candidates."""

# 本地规则重排不发起额外模型请求，结果稳定、可解释，也适合作为外部 reranker 的降级路径。
from __future__ import annotations

import re
from typing import Any

from loguru import logger

from app.rag.models.document import Document
from app.rag.models.search_hit import SearchHit
from app.rag.models.search_query import RetrievalProfile
from app.rag.protocol_metadata import infer_protocol_metadata_from_text
from app.rag.rag_taxonomy import infer_taxonomy_from_text


class LocalReranker:
    """Apply the existing vector, keyword, and metadata weights."""

    # 语义/融合相关度仍是主信号，关键词与 metadata 只做有限纠偏。
    VECTOR_WEIGHT = 0.70
    KEYWORD_WEIGHT = 0.20
    METADATA_WEIGHT = 0.10
    IMPORTANT_RUNBOOK_SECTIONS = {"troubleshooting", "resolution", "verification"}

    def rerank(
        self,
        query: str,
        candidates: list[SearchHit],
        profile: RetrievalProfile,
    ) -> list[SearchHit]:
        ranked: list[SearchHit] = []
        # 每个候选独立计算；单文档 metadata 异常不会让整批排序失败。
        for hit in candidates:
            try:
                vector_score = self._vector_relevance_score(hit.document, hit.score)
                keyword_score = self._keyword_score(query, hit.document)
                metadata_boost = self._metadata_boost(query, hit.document, profile)
                # 三类子分数都归一到非负范围，再按固定权重线性组合。
                final_score = (
                    vector_score * self.VECTOR_WEIGHT
                    + keyword_score * self.KEYWORD_WEIGHT
                    + metadata_boost * self.METADATA_WEIGHT
                )
            except Exception as exc:
                logger.warning(f"候选文档 rerank 失败，使用向量分数: {exc}")
                final_score = self._vector_relevance_score(hit.document, hit.score)

            # 将可观测分数与 profile 写回 metadata，便于 trace 和最终调用方解释结果。
            hit.document.metadata["_retrieval_score"] = final_score
            hit.document.metadata["_retrieval_profile"] = profile.value
            ranked.append(
                SearchHit(document=hit.document, score=final_score, source=hit.source)
            )

        ranked.sort(key=lambda item: item.score, reverse=True)
        return ranked

    def _vector_relevance_score(self, document: Document, raw_score: float) -> float:
        score = max(float(raw_score), 0.0)
        # RRF 已输出“越大越相关”；Milvus 常见距离则用 1/(1+d) 转成同方向相关度。
        if document.metadata.get("_retrieval_score_kind") == "relevance":
            return score
        return 1 / (1 + score)

    def _keyword_score(self, query: str, document: Document) -> float:
        query_terms = self._terms(query)
        if not query_terms:
            return 0.0

        # 除正文外，把高价值结构字段加入词集合，协议地址/告警名不会因正文措辞被遗漏。
        haystack_parts = [document.page_content]
        metadata = document.metadata
        for key in (
            "alert_name",
            "device_type",
            "manufacturer",
            "model",
            "protocol_type",
            "register_address",
            "register_name",
            "function_code",
            "data_type",
            "h1",
            "h2",
            "h3",
            "_file_name",
        ):
            value = metadata.get(key)
            if value:
                haystack_parts.append(str(value))

        haystack_terms = self._terms(" ".join(haystack_parts))
        if not haystack_terms:
            return 0.0

        # 以 query 词数为分母表示“问题有多少部分被候选覆盖”，并把上限截到 1。
        overlap = query_terms & haystack_terms
        return min(len(overlap) / len(query_terms), 1.0)

    def _metadata_boost(
        self,
        query: str,
        document: Document,
        profile: RetrievalProfile,
    ) -> float:
        metadata: dict[str, Any] = document.metadata
        score = 0.0
        query_lower = query.lower()

        # 告警名的完整包含是强证据，因此单项 boost 高于普通标题词命中。
        alert_name = str(metadata.get("alert_name", "")).lower()
        if alert_name and alert_name in query_lower:
            score += 0.5

        section = metadata.get("runbook_section")
        if profile == RetrievalProfile.AIOPS and section in self.IMPORTANT_RUNBOOK_SECTIONS:
            score += 0.3

        # taxonomy/protocol 使用与索引侧相同的规范化函数，比较的是 canonical 值。
        query_taxonomy = infer_taxonomy_from_text(query)
        if query_taxonomy.get("device_type") and query_taxonomy["device_type"] == metadata.get(
            "device_type"
        ):
            score += 0.15
        if query_taxonomy.get("manufacturer") and query_taxonomy[
            "manufacturer"
        ] == metadata.get("manufacturer"):
            score += 0.15

        query_protocol = infer_protocol_metadata_from_text(query)
        if query_protocol.get("register_address") and query_protocol[
            "register_address"
        ] == metadata.get("register_address"):
            score += 0.4
        if query_protocol.get("function_code") and query_protocol["function_code"] == metadata.get(
            "function_code"
        ):
            score += 0.1

        headings = " ".join(str(metadata.get(key, "")) for key in ("h1", "h2", "h3"))
        heading_terms = self._terms(headings)
        if heading_terms and heading_terms & self._terms(query):
            score += 0.2

        # 多个 boost 可叠加，但总 metadata 子分不超过 1，维持权重含义。
        return min(score, 1.0)

    def _terms(self, value: str) -> set[str]:
        lower = value.lower()
        # 英数词按连续 token 提取；中文同时加入单字与相邻双字，兼顾无分词依赖的匹配率。
        terms = set(re.findall(r"[a-z0-9_]+", lower))
        chinese_chars = re.findall(r"[\u4e00-\u9fff]", lower)
        terms.update(chinese_chars)
        terms.update(
            "".join(chinese_chars[index : index + 2])
            for index in range(len(chinese_chars) - 1)
        )
        return {term for term in terms if term.strip()}
