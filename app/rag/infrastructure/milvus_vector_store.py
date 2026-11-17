"""向量存储管理器 - 封装 Milvus VectorStore 操作"""

# 这里是领域 Document/SearchHit 与 LangChain Milvus 类型之间的防腐层。
import inspect
import re
from collections.abc import Sequence
from typing import Any, cast

from langchain_core.documents import Document as LangChainDocument
from langchain_milvus import Milvus
from loguru import logger
from pymilvus import connections

from app.config import config
from app.rag.models.document import Document
from app.rag.models.search_hit import SearchHit
from app.rag.ports.embedding_provider import EmbeddingProvider

# 默认仍使用 biz collection，可通过环境变量 MILVUS_COLLECTION_NAME 覆盖
COLLECTION_NAME = config.milvus_collection_name
METADATA_KEY_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
# key 的白名单正则防止调用方把任意表达式片段注入 Milvus metadata 过滤语句。


class MilvusVectorStore:
    """向量存储管理器"""

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        *,
        milvus_cls: type[Milvus] = Milvus,
        connections_api: Any = connections,
    ) -> None:
        """初始化向量存储管理器"""
        # milvus_cls 与 connections_api 可注入，既支持测试替身，也保留兼容层的 patch 能力。
        self.vector_store = None
        self.collection_name = COLLECTION_NAME
        self.embedding_provider = embedding_provider
        self._milvus_cls = milvus_cls
        self._connections_api = connections_api
        self._initialize_vector_store()

    def _initialize_vector_store(self):
        """初始化 Milvus VectorStore"""
        try:
            # connection_args 同时交给低层连接与 LangChain wrapper，确保数据库和 collection 上下文一致。
            connection_args = {
                "host": config.milvus_host,
                "port": config.milvus_port,
                "db_name": config.milvus_database_name,
            }
            self._connections_api.connect(
                alias=f"http://{config.milvus_host}:{config.milvus_port}",
                **connection_args,
            )

            # 创建 LangChain Milvus VectorStore
            # 使用 biz collection，字段映射：text_field -> content, vector_field -> vector
            # auto_id=False 配合稳定 _chunk_id，使重试不会为相同 chunk 随机生成第二条记录。
            self.vector_store = self._milvus_cls(
                embedding_function=cast(Any, self.embedding_provider),
                collection_name=self.collection_name,
                connection_args=connection_args,
                auto_id=False,  # 使用自定义 id
                drop_old=False,
                text_field="content",  # 文本内容存储到 content 字段
                vector_field="vector",  # 向量存储到 vector 字段
                primary_field="id",  # 主键字段
                metadata_field="metadata",  # 元数据字段
            )

            logger.info(
                f"VectorStore 初始化成功: {config.milvus_host}:{config.milvus_port}, "
                f"database: {config.milvus_database_name}, collection: {self.collection_name}"
            )

        except Exception as e:
            logger.error(f"VectorStore 初始化失败: {e}")
            raise

    def add_documents(self, documents: Sequence[Document]) -> list[str]:
        """
        批量添加文档到向量存储（自动批量向量化）

        Args:
            documents: 文档列表

        Returns:
            List[str]: 文档 ID 列表
        """
        try:
            import time
            import uuid

            start_time = time.time()

            # 优先使用稳定 chunk id；旧调用未提供时回退到 uuid
            # 新索引路径使用稳定 chunk id；UUID 仅为没有新版 metadata 的旧调用保底。
            ids = [doc.metadata.get("_chunk_id") or str(uuid.uuid4()) for doc in documents]

            # 在适配器边界复制 metadata，防止 LangChain 内部处理反向修改领域对象。
            langchain_documents = [
                LangChainDocument(
                    page_content=document.page_content,
                    metadata=dict(document.metadata),
                    **({"id": document.id} if getattr(document, "id", None) else {}),
                )
                for document in documents
            ]

            # LangChain Milvus 的 add_documents 会自动调用 embedding_function
            # 并进行批量处理，性能更好
            # embedding_provider 会由 LangChain 自动调用，IndexingService 无需接触向量 SDK 类型。
            result_ids = self.vector_store.add_documents(langchain_documents, ids=ids)

            elapsed = time.time() - start_time
            logger.info(
                f"批量添加 {len(documents)} 个文档到 VectorStore 完成, "
                f"耗时: {elapsed:.2f}秒, 平均: {elapsed / len(documents):.2f}秒/个"
            )
            return result_ids
        except Exception as e:
            logger.error(f"添加文档失败: {e}")
            raise

    def delete_by_source(self, file_path: str) -> int:
        """
        删除指定文件的所有文档

        Args:
            file_path: 文件路径

        Returns:
            int: 删除的文档数量
        """
        try:
            # 使用 milvus_manager 获取已连接的 collection
            from app.rag.milvus_client import milvus_manager

            # 删除/查询 generation 使用底层 Collection，因为 LangChain wrapper 没有完整暴露这些操作。
            collection = milvus_manager.get_collection()

            deleted_count = self._delete_by_metadata("_source", file_path, collection=collection)
            logger.info(f"删除文件旧数据: {file_path}, 删除数量: {deleted_count}")
            return deleted_count

        except Exception as e:
            logger.warning(f"删除旧数据失败 (可能是首次索引): {e}")
            return 0

    def get_legacy_ids_by_source(self, source: str) -> list[str]:
        """Capture legacy primary IDs without including stable generations."""
        from app.rag.milvus_client import milvus_manager

        collection = milvus_manager.get_collection()
        expr = f'metadata["_source"] == "{self._escape_expr_value(source)}"'
        # 先按旧 _source 找候选，再排除已有 _file_id 的新 generation，避免误删新版记录。
        rows = collection.query(expr=expr, output_fields=["id", "metadata"])
        return [
            str(row["id"])
            for row in rows
            if row.get("id") is not None
            and isinstance(row.get("metadata"), dict)
            and "_file_id" not in row["metadata"]
        ]

    def delete_by_ids(self, document_ids: list[str]) -> int:
        """Delete only the captured primary IDs."""
        # dict 保序去重，既缩短表达式也让日志/测试中的删除顺序稳定。
        unique_ids = list(dict.fromkeys(str(document_id) for document_id in document_ids))
        if not unique_ids:
            return 0
        quoted_ids = ", ".join(
            f'"{self._escape_expr_value(document_id)}"' for document_id in unique_ids
        )
        return self._delete_by_expr(f"id in [{quoted_ids}]")

    def has_file_id(self, file_id: str) -> bool:
        """判断指定 file_id 是否已存在。"""
        return self._has_metadata("_file_id", file_id)

    def has_file_hash(self, file_id: str, file_hash: str) -> bool:
        """判断指定 file_id + file_hash 是否已存在。"""
        from app.rag.milvus_client import milvus_manager

        collection = milvus_manager.get_collection()
        expr = (
            f'metadata["_file_id"] == "{self._escape_expr_value(file_id)}" '
            f'&& metadata["_file_hash"] == "{self._escape_expr_value(file_hash)}"'
        )
        # 只需知道 generation 是否存在，limit=1 避免返回该文件的全部 chunk。
        rows = collection.query(expr=expr, output_fields=["id"], limit=1)
        return bool(rows)

    def delete_by_file_id(self, file_id: str) -> int:
        """按稳定 file_id 删除文档。"""
        return self._delete_by_metadata("_file_id", file_id)

    def delete_by_file_hash(self, file_hash: str) -> int:
        """按 file_hash 删除文档。"""
        return self._delete_by_metadata("_file_hash", file_hash)

    def delete_by_file_generation(self, file_id: str, file_hash: str) -> int:
        """Delete one staged generation without touching older file content."""
        # 精确 generation 删除用于回滚本次部分写入，不触碰旧的可用版本。
        expr = (
            f'metadata["_file_id"] == "{self._escape_expr_value(file_id)}" '
            f'&& metadata["_file_hash"] == "{self._escape_expr_value(file_hash)}"'
        )
        return self._delete_by_expr(expr)

    def delete_file_generations(self, file_id: str, keep_file_hash: str) -> int:
        """Delete obsolete generations after the replacement is fully written."""
        # replace 完成后以 != 保留当前 hash，一次清除该路径的所有过期版本。
        expr = (
            f'metadata["_file_id"] == "{self._escape_expr_value(file_id)}" '
            f'&& metadata["_file_hash"] != "{self._escape_expr_value(keep_file_hash)}"'
        )
        return self._delete_by_expr(expr)

    def _has_metadata(self, key: str, value: str) -> bool:
        from app.rag.milvus_client import milvus_manager

        collection = milvus_manager.get_collection()
        expr = f'metadata["{key}"] == "{self._escape_expr_value(value)}"'
        rows = collection.query(expr=expr, output_fields=["id"], limit=1)
        return bool(rows)

    def _delete_by_metadata(self, key: str, value: str, collection=None) -> int:
        expr = f'metadata["{key}"] == "{self._escape_expr_value(value)}"'
        return self._delete_by_expr(expr, collection=collection)

    def _delete_by_expr(self, expr: str, collection=None) -> int:
        from app.rag.milvus_client import milvus_manager

        # 可传入已取得的 collection，避免同一操作链重复解析全局 manager。
        target_collection = collection or milvus_manager.get_collection()
        result = target_collection.delete(expr)
        return int(getattr(result, "delete_count", 0) or 0)

    def _escape_expr_value(self, value: str) -> str:
        # 先转义反斜线再转义双引号，保证值始终留在字符串字面量内部。
        return value.replace("\\", "\\\\").replace('"', '\\"')

    def _metadata_equals_expr(self, key: str, value: str) -> str:
        """Build a Milvus JSON metadata equality expression."""
        return f'metadata["{key}"] == "{self._escape_expr_value(str(value))}"'

    def _metadata_filter_expr(
        self,
        metadata_filter: dict[str, str | list[str]] | None,
    ) -> str:
        """Build a safe Milvus expression for simple metadata equality filters."""
        if not metadata_filter:
            return ""

        # 每个字典项用 AND 连接；列表值则在同一字段内用 OR 连接。
        clauses: list[str] = []
        for key, value in metadata_filter.items():
            if not key or value is None or value == "":
                continue
            # value 会经过字符串转义，key 则必须完整匹配标识符白名单。
            if not self._is_safe_metadata_key(key):
                raise ValueError(f"unsafe metadata filter key: {key}")

            if isinstance(value, list):
                # 空列表不能表达有效过滤，直接忽略；非空列表展开为括号内析取式。
                equals = [
                    self._metadata_equals_expr(key, item)
                    for item in value
                    if item is not None and item != ""
                ]
                if not equals:
                    continue
                clauses.append(f"({' || '.join(equals)})" if len(equals) > 1 else equals[0])
                continue

            clauses.append(self._metadata_equals_expr(key, value))

        return " && ".join(clauses)

    def _is_safe_metadata_key(self, key: str) -> bool:
        return bool(METADATA_KEY_PATTERN.fullmatch(key))

    def get_vector_store(self) -> Milvus:
        """
        获取 VectorStore 实例

        Returns:
            Milvus: VectorStore 实例
        """
        return cast(Milvus, self.vector_store)

    def similarity_search(self, query: str, k: int = 3) -> list[LangChainDocument]:
        """
        相似度搜索

        Args:
            query: 查询文本
            k: 返回结果数量

        Returns:
            List[Document]: 相关文档列表
        """
        try:
            docs = self.vector_store.similarity_search(query, k=k)
            logger.debug(f"相似度搜索完成: query='{query}', 结果数={len(docs)}")
            return docs
        except Exception as e:
            logger.error(f"相似度搜索失败: {e}")
            return []

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 3,
        metadata_filter: dict[str, str | list[str]] | None = None,
    ) -> list[tuple[LangChainDocument, float]]:
        """带分数的相似度搜索。"""
        # 过滤器构造失败时拒绝查询，不能悄悄退回无过滤召回造成数据域越界。
        try:
            expr = self._metadata_filter_expr(metadata_filter)
        except ValueError as exc:
            logger.error(f"metadata filter 非法，拒绝查询: {exc}")
            return []
        if metadata_filter and not expr:
            logger.error("metadata filter 未生成安全表达式，拒绝未过滤查询")
            return []
        try:
            # 只有合法且非空的表达式才下推 expr；无 filter 时保持普通相似度检索。
            if expr:
                docs = self._call_similarity_search_with_score(query, k=k, expr=expr)
            else:
                docs = self._call_similarity_search_with_score(query, k=k)
            logger.debug(f"带分数相似度搜索完成: query='{query}', 结果数={len(docs)}")
            return docs
        except Exception as e:
            # 带过滤查询执行失败也不做无过滤回退，这是查询正确性与隔离性的硬边界。
            if expr:
                logger.error(f"带分数相似度过滤搜索失败，拒绝未过滤回退: {e}")
                return []
            logger.error(f"带分数相似度搜索失败: {e}")
            return []

    def search(
        self,
        query: str,
        k: int,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[SearchHit]:
        """Return dense candidates using framework-neutral models."""
        # 在端口出口把 LangChain Document 转回领域对象，查询用例不感知具体 vector store。
        return [
            SearchHit(
                document=Document(
                    page_content=document.page_content,
                    metadata=dict(document.metadata),
                    id=getattr(document, "id", None),
                ),
                score=float(score),
                source="dense",
            )
            for document, score in self.similarity_search_with_score(
                query,
                k=k,
                metadata_filter=metadata_filter,
            )
        ]

    def supports_metadata_filter(self) -> bool:
        """Milvus filtering is part of this adapter's stable contract."""
        return True

    def _call_similarity_search_with_score(
        self,
        query: str,
        k: int,
        expr: str | None = None,
    ) -> list[tuple[LangChainDocument, float]]:
        kwargs: dict[str, Any] = {"k": k}
        if expr:
            kwargs["expr"] = expr
        # 不同 langchain-milvus 版本的签名不同，仅在兼容时注入底层搜索参数。
        if self._vector_store_accepts_search_param():
            kwargs["param"] = self._search_params()
        return self.vector_store.similarity_search_with_score(query, **kwargs)

    def _search_params(self) -> dict[str, Any]:
        # metric_type 必须与 collection 索引一致；nprobe 控制召回质量与扫描成本。
        return {
            "metric_type": config.rag_vector_metric_type,
            "params": {"nprobe": config.rag_vector_nprobe},
        }

    def _vector_store_accepts_search_param(self) -> bool:
        method = getattr(self.vector_store, "similarity_search_with_score", None)
        if method is None:
            return False

        # 对不可 introspect 的代理/扩展方法采取兼容性乐观判断，让其自行接收 kwargs。
        try:
            signature = inspect.signature(method)
        except (TypeError, ValueError):
            return True

        return "param" in signature.parameters or any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        )

    def optimize_collection(self) -> dict[str, Any]:
        """Best-effort 优化 collection/index。"""
        # 优化不影响索引事务结论，故所有步骤用状态对象汇报并在异常时降级为 warning。
        status: dict[str, Any] = {
            "optimized": False,
            "flushed": False,
            "compacted": False,
        }
        try:
            from app.rag.milvus_client import milvus_manager

            collection = milvus_manager.get_collection()
            if hasattr(collection, "flush"):
                collection.flush()
                status["flushed"] = True
            if hasattr(collection, "compact"):
                collection.compact()
                status["compacted"] = True
            status["optimized"] = True
            return status
        except Exception as e:
            logger.warning(f"Collection 优化失败: {e}")
            status["error"] = str(e)
            return status
