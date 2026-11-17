"""Metadata builder for RAG document chunks."""

# metadata builder 是 chunk 身份与过滤 schema 的统一出口，避免各切分策略各自造字段。
from __future__ import annotations

import hashlib
from collections.abc import Iterable
from datetime import UTC, datetime

from app.rag.indexing.contracts import LoadedDocument
from app.rag.models.document import Document
from app.rag.protocol_metadata import normalize_protocol_metadata
from app.rag.rag_taxonomy import (
    infer_taxonomy_from_document,
    normalize_taxonomy_metadata,
    route_key_with_taxonomy,
)


class RagMetadataBuilder:
    """Add stable file and chunk metadata before vector writes."""

    def __init__(
        self,
        schema_version: int | None = None,
        collection_name: str | None = None,
    ) -> None:
        # 显式参数便于测试；缺省时才延迟读取应用配置，降低模块导入副作用。
        if schema_version is None or not collection_name:
            from app.config import config

            if schema_version is None:
                schema_version = config.rag_schema_version
            if not collection_name:
                collection_name = config.milvus_collection_name
        self.schema_version = schema_version
        self.collection_name = collection_name

    def build(
        self,
        loaded: LoadedDocument,
        documents: Iterable[Document],
        parser: str,
    ) -> list[Document]:
        # 先物化 Iterable 才能确定 _chunk_count，并保证 enumerate 的顺序就是稳定 chunk 顺序。
        docs = list(documents)
        chunk_count = len(docs)
        # 同一批 chunk 共享时间戳，表示它们属于一次文件 generation 写入。
        indexed_at = datetime.now(UTC).isoformat()
        enriched: list[Document] = []

        for chunk_index, doc in enumerate(docs):
            # chunk_id 同时绑定文件身份、内容 generation、顺序与正文，可直接作为稳定存储主键。
            chunk_hash = self._sha256_text(doc.page_content)
            chunk_id = self._sha256_text(
                f"{loaded.file_id}:{loaded.file_hash}:{chunk_index}:{chunk_hash}"
            )
            # frontmatter 是底层默认值，切分器产出的 chunk metadata 可覆盖同名字段。
            metadata = dict(getattr(loaded, "metadata", {}) or {})
            metadata.update(doc.metadata)
            self._normalize_filter_metadata(metadata, loaded, doc.page_content)
            # 下划线字段记录索引事实，业务过滤字段则已在 normalize 阶段补齐。
            metadata.update(
                {
                    "_schema_version": self.schema_version,
                    "_collection_name": self.collection_name,
                    "_source": loaded.normalized_path,
                    "_file_id": loaded.file_id,
                    "_file_hash": loaded.file_hash,
                    "_chunk_id": chunk_id,
                    "_chunk_hash": chunk_hash,
                    "_chunk_index": chunk_index,
                    "_chunk_count": chunk_count,
                    "_content_length": len(doc.page_content),
                    "_parser": parser,
                    "_indexed_at": indexed_at,
                    "_file_name": loaded.file_name,
                    "_extension": loaded.extension,
                    "_file_size": loaded.file_size,
                    "_modified_time": loaded.modified_time,
                }
            )
            # 返回新的领域 Document，避免就地修改上游切分结果造成阶段间隐式共享状态。
            enriched.append(Document(page_content=doc.page_content, metadata=metadata))

        return enriched

    def _normalize_filter_metadata(
        self,
        metadata: dict,
        loaded: LoadedDocument,
        content: str = "",
    ) -> None:
        # 先规范用户/切分器已有值，再用文件名和正文推断缺失值，最后再次统一推断结果格式。
        normalize_taxonomy_metadata(metadata)
        inferred_taxonomy = infer_taxonomy_from_document(loaded.file_name, metadata, content)
        for key, value in inferred_taxonomy.items():
            metadata.setdefault(key, value)
        normalize_taxonomy_metadata(metadata)
        # 协议字段与 taxonomy 一并在写入前固化，查询时可以直接下推 metadata filter。
        normalize_protocol_metadata(metadata, file_name=loaded.file_name, content=content)

        # 显式 document_type 优先；没有时按 runbook_section 判断是否属于运维手册。
        if "document_type" not in metadata:
            runbook_section = str(metadata.get("runbook_section", "")).strip().lower()
            metadata["document_type"] = (
                "runbook" if runbook_section and runbook_section != "other" else "document"
            )

        # heading_path 把分层标题压成可检索、可展示的单一导航路径。
        if "heading_path" not in metadata:
            headings = [
                str(metadata[key]).strip() for key in ("h1", "h2", "h3") if metadata.get(key)
            ]
            if headings:
                metadata["heading_path"] = " > ".join(headings)

        # route_key 从最具体的协议/告警字段逐级回退，最终至少可以用文件名路由。
        if "route_key" not in metadata:
            base_route_key = (
                metadata.get("register_address")
                or metadata.get("alert_name")
                or metadata.get("heading_path")
                or loaded.file_name
            )
            metadata["route_key"] = route_key_with_taxonomy(metadata, str(base_route_key))

    def _sha256_text(self, value: str) -> str:
        return hashlib.sha256(value.encode("utf-8")).hexdigest()
