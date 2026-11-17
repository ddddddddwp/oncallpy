"""Deduplication decisions for incremental RAG indexing."""

# 去重只读取向量存储中的 generation 事实；真正的写入和安全清理仍由 IndexingService 编排。
from __future__ import annotations

from loguru import logger

from app.rag.indexing.contracts import IngestionDecision, LoadedDocument
from app.rag.ports.vector_writer import VectorWriter


class RagDeduplicator:
    """Decide whether a file should be skipped, inserted, or replaced."""

    def __init__(self, vector_store: VectorWriter) -> None:
        self.vector_store = vector_store

    def prepare(self, loaded: LoadedDocument) -> IngestionDecision:
        # 先捕获旧版 _source 数据，后续无论 skip/insert/replace 都可在安全时机清理。
        legacy_ids = self._get_legacy_ids_by_source(loaded.normalized_path)
        # file_id + file_hash 命中表示相同路径的相同内容已存在，dense 无需重复写入。
        if self._has_file_hash(loaded.file_id, loaded.file_hash):
            return IngestionDecision(
                action="skip",
                reason="unchanged",
                legacy_ids=legacy_ids,
            )

        # 仅 file_id 命中说明路径相同但内容变了：新旧数据属于不同 generation。
        if self._has_file_id(loaded.file_id):
            return IngestionDecision(
                action="replace",
                reason="file_hash_changed",
                legacy_ids=legacy_ids,
                can_rollback_generation=True,
            )

        # 没有稳定 file_id 时是首次写入；若发现 legacy id，则本次也承担旧格式迁移。
        return IngestionDecision(
            action="insert",
            reason="legacy_source_replaced" if legacy_ids else "new_file",
            legacy_ids=legacy_ids,
            can_rollback_generation=True,
        )

    def _has_file_hash(self, file_id: str, file_hash: str) -> bool:
        return bool(self.vector_store.has_file_hash(file_id, file_hash))

    def _has_file_id(self, file_id: str) -> bool:
        return bool(self.vector_store.has_file_id(file_id))

    def _get_legacy_ids_by_source(self, source: str) -> list[str]:
        # getattr 保持对旧 VectorWriter 实现的兼容：没有迁移能力时仍可完成正常索引。
        get_legacy_ids = getattr(self.vector_store, "get_legacy_ids_by_source", None)
        if get_legacy_ids is None:
            return []
        try:
            return list(get_legacy_ids(source))
        # 查询 legacy 数据失败不是新 generation 写入的硬失败，记录告警后继续。
        except Exception as exc:
            logger.warning(f"查询 legacy source 数据失败，将继续索引新数据: {exc}")
            return []
