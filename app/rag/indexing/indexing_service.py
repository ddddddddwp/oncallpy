"""向量索引服务模块"""

# 该服务只负责编排；文件解析、身份判断、metadata 和存储操作均通过注入的协作者完成。
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger

from app.config import config
from app.rag.indexing.contracts import (
    Deduplicator,
    DocumentLoader,
    DocumentSplitter,
    FileIndexingOutcome,
    LoadedDocument,
    MetadataBuilder,
)
from app.rag.models.document import Document
from app.rag.models.indexing_result import IndexingResult
from app.rag.ports.sparse_writer import SparseWriter
from app.rag.ports.vector_writer import VectorWriter
from app.rag.protocol_metadata import normalize_protocol_metadata
from app.rag.rag_taxonomy import (
    infer_taxonomy_from_document,
    normalize_taxonomy_metadata,
    route_key_with_taxonomy,
)


class IndexingService:
    """向量索引服务 - 负责读取文件、生成向量、存储到 Milvus"""

    def __init__(
        self,
        *,
        loader: DocumentLoader,
        splitter: DocumentSplitter,
        metadata_builder: MetadataBuilder,
        deduplicator: Deduplicator,
        vector_writer: VectorWriter,
        sparse_writer: SparseWriter | None = None,
        sparse_writer_factory: Callable[[], SparseWriter | None] | None = None,
    ) -> None:
        """初始化向量索引服务"""
        # 默认目录属于用例配置，而外部资源则全部以端口/factory 注入。
        self.upload_path = "./uploads"
        self.loader = loader
        self.splitter = splitter
        self.metadata_builder = metadata_builder
        self.vector_writer = vector_writer
        self.sparse_writer = sparse_writer
        self.sparse_writer_factory = sparse_writer_factory
        self.deduplicator = deduplicator
        logger.info("向量索引服务初始化完成")

    def index_directory(self, directory_path: str | None = None) -> IndexingResult:
        """
        索引指定目录下的所有文件

        Args:
            directory_path: 目录路径（可选，默认使用配置的上传目录）

        Returns:
            IndexingResult: 索引结果
        """
        # 即使目录级验证或枚举失败，也返回结构完整的 IndexingResult，而不是让批任务丢失统计。
        result = IndexingResult()
        result.start_time = datetime.now()

        try:
            # 使用指定目录或默认上传目录
            target_path = directory_path if directory_path else self.upload_path
            dir_path = Path(target_path).resolve()

            if not dir_path.exists() or not dir_path.is_dir():
                raise ValueError(f"目录不存在或不是有效目录: {target_path}")

            result.directory_path = str(dir_path)

            # 获取所有支持的文件
            # glob 不递归：目录索引的边界明确限定为第一层受支持文本文件。
            files = list(dir_path.glob("*.txt")) + list(dir_path.glob("*.md"))

            if not files:
                logger.warning(f"目录中没有找到支持的文件: {target_path}")
                result.total_files = 0
                result.success = True
                result.end_time = datetime.now()
                return result

            result.total_files = len(files)
            logger.info(f"开始索引目录: {target_path}, 找到 {len(files)} 个文件")

            # 遍历并索引每个文件
            # 每个文件独立捕获异常，一个坏文件不会阻断同目录其他文件。
            for file_path in files:
                try:
                    outcome = self.index_single_file(str(file_path))
                    if outcome.skipped:
                        result.increment_skipped_count()
                        logger.info(f"✓ 文件未变化，跳过索引: {file_path.name}")
                    else:
                        result.increment_success_count()
                        logger.info(f"✓ 文件索引成功: {file_path.name}")
                    result.deleted_count += outcome.deleted_count
                    result.inserted_chunk_count += outcome.inserted_chunks
                except Exception as e:
                    result.increment_fail_count()
                    result.add_failed_file(str(file_path), str(e))
                    logger.error(f"✗ 文件索引失败: {file_path.name}, 错误: {e}")

            result.success = result.fail_count == 0
            result.end_time = datetime.now()

            logger.info(
                f"目录索引完成: 总数={result.total_files}, "
                f"成功={result.success_count}, 失败={result.fail_count}"
            )

            return result

        except Exception as e:
            logger.error(f"索引目录失败: {e}")
            result.success = False
            result.error_message = str(e)
            result.end_time = datetime.now()
            return result

    def index_single_file(self, file_path: str):
        """
        索引单个文件 (使用新的 LangChain 分割器)

        Args:
            file_path: 文件路径

        Raises:
            ValueError: 文件不存在时抛出
            RuntimeError: 索引失败时抛出
        """
        path = Path(file_path).resolve()

        if not path.exists() or not path.is_file():
            raise ValueError(f"文件不存在: {file_path}")

        logger.info(f"开始索引文件: {path}")

        try:
            # 1. 加载阶段同时得到路径身份 file_id 与内容身份 file_hash。
            loaded = self.loader.load(str(path))
            logger.info(f"读取文件: {path}, 内容长度: {len(loaded.text)} 字符")

            # 2. dense 存储事实决定 skip/insert/replace，但 sparse 完整性还要单独检查。
            decision = self.deduplicator.prepare(loaded)
            outcome = FileIndexingOutcome(
                file_path=loaded.normalized_path,
                file_id=loaded.file_id,
                file_hash=loaded.file_hash,
                parser=self._select_parser(loaded.extension),
                skipped=decision.skipped,
                reason=decision.reason,
                deleted_count=decision.deleted_count,
            )

            # dense 已有而 sparse 缺失时进入修复模式：复用切分结果，但绝不重复写 dense。
            repair_sparse = decision.skipped and not self._sparse_generation_present(loaded)
            if decision.skipped and not repair_sparse:
                # 即使内容未变，也可清理同 file_id 的旧 generation 与 legacy 主键。
                outcome.deleted_count += self.vector_writer.delete_file_generations(
                    loaded.file_id,
                    loaded.file_hash,
                )
                if decision.legacy_ids:
                    outcome.deleted_count += self.vector_writer.delete_by_ids(decision.legacy_ids)
                logger.info(f"文件内容未变化，跳过索引: {file_path}")
                return outcome
            if repair_sparse:
                outcome.skipped = False
                outcome.reason = "sparse_repaired"

            # 3. 先生成内容 chunk，再追加轻量 route summary；两者一起进入统一 metadata builder。
            documents = self.splitter.split_document(loaded.text, loaded.normalized_path)
            documents.extend(self._build_route_summary_documents(loaded, documents))
            logger.info(f"文档分割完成: {file_path} -> {len(documents)} 个分片")

            # 4. 在写存储前一次性固化稳定 chunk id、generation 和可过滤业务字段。
            enriched_documents = self.metadata_builder.build(
                loaded=loaded,
                documents=documents,
                parser=outcome.parser,
            )

            if not enriched_documents:
                raise ValueError("no indexable chunks")

            # 5. 正常模式先写 dense；失败时仅回滚本次 generation，旧 generation 尚未删除。
            if not repair_sparse:
                try:
                    outcome.document_ids = self.vector_writer.add_documents(enriched_documents)
                except Exception:
                    if decision.can_rollback_generation:
                        self._rollback_dense_generation(loaded)
                    raise

            outcome.inserted_chunks = len(enriched_documents)
            # 6. dense 成功后再用事务替换 sparse；只有两路完成后才清理旧 dense 数据。
            self._sync_sparse_index(loaded, enriched_documents)
            if decision.action == "replace" or repair_sparse:
                outcome.deleted_count += self.vector_writer.delete_file_generations(
                    loaded.file_id,
                    loaded.file_hash,
                )
            if decision.legacy_ids:
                outcome.deleted_count += self.vector_writer.delete_by_ids(decision.legacy_ids)
            # compact/flush 属于 best-effort 后处理，其结果单独记录而不改变已完成的索引事实。
            outcome.optimization = self.vector_writer.optimize_collection()
            logger.info(f"文件索引完成: {file_path}, 共 {len(enriched_documents)} 个分片")

            return outcome

        except Exception as e:
            logger.error(f"索引文件失败: {file_path}, 错误: {e}")
            raise RuntimeError(f"索引文件失败: {e}") from e

    def _sync_sparse_index(
        self,
        loaded: LoadedDocument,
        documents: list[Document],
    ) -> None:
        """Write successfully inserted chunks to the sparse index."""
        # sparse 不可用时保持既有 dense-only 降级；适配器可用时优先走整文件原子替换。
        sparse_index = self._get_sparse_index()
        if sparse_index is None:
            return
        # getattr 兼容只实现旧 upsert_documents 接口的 SparseWriter。
        replace = getattr(sparse_index, "replace_file_documents", None)
        if replace is not None:
            replace(loaded.file_id, documents)
            return
        sparse_index.upsert_documents(documents)

    def _sparse_generation_present(self, loaded: LoadedDocument) -> bool:
        # 没有 sparse 配置代表系统主动运行在 dense-only，而不是需要不断“修复”的缺失状态。
        sparse_index = self._get_sparse_index()
        if sparse_index is None:
            return True
        # 旧 sparse 实现无法检查 generation 时视为已满足，保持向后兼容。
        contains = getattr(sparse_index, "contains_file_hash", None)
        if contains is None:
            return True
        try:
            return bool(contains(loaded.file_id, loaded.file_hash))
        # 检查异常按“可能缺失”处理，主动重建比静默留下不一致更安全。
        except Exception as exc:
            logger.warning(f"检查 sparse generation 失败，将尝试修复: {exc}")
            return False

    def _rollback_dense_generation(self, loaded: LoadedDocument) -> None:
        """Best-effort rollback for a partially written staged generation."""
        # 回滚只针对本次 file_id + file_hash，不会误删仍可服务的旧内容版本。
        rollback = getattr(self.vector_writer, "delete_by_file_generation", None)
        if rollback is None:
            return
        try:
            rollback(loaded.file_id, loaded.file_hash)
        except Exception as exc:
            logger.warning(f"回滚未完成的 dense generation 失败: {exc}")

    def _get_sparse_index(self):
        # 显式实例优先于 factory，方便测试注入并避免重复解析外部资源。
        if self.sparse_writer is not None:
            return self.sparse_writer
        if self.sparse_writer_factory is None:
            return None
        # factory 失败是可降级事件；主索引仍可通过 dense 提供服务。
        try:
            return self.sparse_writer_factory()
        except Exception as exc:
            logger.warning(f"稀疏索引不可用，跳过同步: {exc}")
            return None

    def _select_parser(self, extension: str) -> str:
        """根据扩展名选择 parser 名称。"""
        return "markdown" if extension.lower() == ".md" else "text"

    def _build_route_summary_documents(
        self, loaded: LoadedDocument, documents: list[Document]
    ) -> list[Document]:
        """Build lightweight route summaries used by route-first retrieval."""
        # 关闭 route-first 时完全不生成摘要，查询层会直接执行内容召回。
        if not config.rag_route_retrieval_enabled:
            return []

        route_docs: list[Document] = []
        seen_route_keys: set[str] = set()
        for doc in documents:
            # 复刻 metadata builder 的标准化顺序，使摘要 route_key 与最终内容 chunk 一致。
            metadata = dict(getattr(loaded, "metadata", {}) or {})
            metadata.update(doc.metadata)
            normalize_taxonomy_metadata(metadata)
            inferred_taxonomy = infer_taxonomy_from_document(
                loaded.file_name,
                metadata,
                doc.page_content,
            )
            for key, value in inferred_taxonomy.items():
                metadata.setdefault(key, value)
            normalize_taxonomy_metadata(metadata)
            normalize_protocol_metadata(
                metadata,
                file_name=loaded.file_name,
                content=doc.page_content,
            )
            # 防御性跳过已有摘要，避免摘要再派生摘要形成重复路由文档。
            if metadata.get("document_type") == "route_summary":
                continue

            route_key = self._route_key(metadata, loaded)
            # 同一文件的相同 route_key 只保留一条轻量摘要，降低第一阶段召回噪声。
            if not route_key or route_key in seen_route_keys:
                continue

            seen_route_keys.add(route_key)
            target_document_type = self._target_document_type(metadata)
            route_metadata = {
                "document_type": "route_summary",
                "target_document_type": target_document_type,
                "route_key": route_key,
                "_source": loaded.normalized_path,
                "_file_name": loaded.file_name,
            }
            # 只复制真正参与路由/展示的字段，不携带完整 chunk generation metadata。
            for key in (
                "alert_name",
                "business_domain",
                "device_type",
                "manufacturer",
                "model",
                "protocol_type",
                "register_address",
                "register_name",
                "function_code",
                "data_type",
                "unit",
                "scale",
                "read_write",
                "_source",
                "_file_name",
                "h1",
                "h2",
                "h3",
            ):
                if metadata.get(key):
                    route_metadata[key] = metadata[key]

            route_docs.append(
                Document(
                    page_content=self._route_summary_text(route_metadata),
                    metadata=route_metadata,
                )
            )

        return route_docs

    def _route_key(self, metadata: dict[str, Any], loaded: LoadedDocument) -> str:
        # 候选从显式 route_key 逐级退化到文件名，确保普通文档也能建立路由入口。
        heading_path = " > ".join(
            str(metadata[key]).strip() for key in ("h1", "h2", "h3") if metadata.get(key)
        )
        value = (
            metadata.get("route_key")
            or metadata.get("register_address")
            or metadata.get("alert_name")
            or metadata.get("business_domain")
            or heading_path
            or loaded.file_name
        )
        if not value:
            return ""
        # taxonomy 前缀让相同标题在不同设备/厂商域下仍能形成不同路由键。
        return route_key_with_taxonomy(metadata, str(value))

    def _target_document_type(self, metadata: dict[str, Any]) -> str:
        # 摘要自身类型永远是 route_summary，这里推断的是第二阶段应检索的内容类型。
        document_type = metadata.get("document_type")
        if document_type and document_type != "route_summary":
            return str(document_type)

        if metadata.get("register_address") or metadata.get("protocol_type"):
            return "protocol"

        section = str(metadata.get("runbook_section", "")).strip().lower()
        if section and section != "other":
            return "runbook"
        return "document"

    def _route_summary_text(self, metadata: dict[str, Any]) -> str:
        # 采用稳定的 key:value 文本，使 embedding 与 sparse 检索都能命中结构字段和值。
        parts = []
        if metadata.get("_file_name"):
            parts.append(f"file_name: {metadata['_file_name']}")
        parts.append(f"route_key: {metadata['route_key']}")
        if metadata.get("alert_name"):
            parts.append(f"alert_name: {metadata['alert_name']}")
        if metadata.get("business_domain"):
            parts.append(f"business_domain: {metadata['business_domain']}")
        if metadata.get("device_type"):
            parts.append(f"device_type: {metadata['device_type']}")
        if metadata.get("manufacturer"):
            parts.append(f"manufacturer: {metadata['manufacturer']}")
        if metadata.get("model"):
            parts.append(f"model: {metadata['model']}")
        if metadata.get("protocol_type"):
            parts.append(f"protocol_type: {metadata['protocol_type']}")
        if metadata.get("register_address"):
            parts.append(f"register_address: {metadata['register_address']}")
        if metadata.get("register_name"):
            parts.append(f"register_name: {metadata['register_name']}")
        if metadata.get("function_code"):
            parts.append(f"function_code: {metadata['function_code']}")
        if metadata.get("data_type"):
            parts.append(f"data_type: {metadata['data_type']}")
        if metadata.get("unit"):
            parts.append(f"unit: {metadata['unit']}")
        if metadata.get("scale"):
            parts.append(f"scale: {metadata['scale']}")
        if metadata.get("read_write"):
            parts.append(f"read_write: {metadata['read_write']}")
        # 标题合并为单条路径，保留章节层级但避免摘要正文过长。
        headings = [str(metadata[key]) for key in ("h1", "h2", "h3") if metadata.get(key)]
        if headings:
            parts.append(f"heading_path: {' > '.join(headings)}")
        parts.append(f"target_document_type: {metadata['target_document_type']}")
        return "\n".join(parts)
