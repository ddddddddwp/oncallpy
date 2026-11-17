"""SQLite FTS sparse index for exact-term RAG retrieval."""

# sparse 通道擅长精确关键词、寄存器地址和型号，与语义 dense 召回形成互补。
from __future__ import annotations

import json
import re
import sqlite3
from collections.abc import Generator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from langchain_core.documents import Document as LangChainDocument
from loguru import logger

from app.config import config
from app.rag.models.document import Document
from app.rag.models.search_hit import SearchHit

TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_.:/#-]+|[\u4e00-\u9fff]+")
# 允许的 metadata 字段同时充当过滤白名单，禁止动态拼接任意 JSON path。
METADATA_FIELDS = (
    "document_type",
    "alert_name",
    "business_domain",
    "runbook_section",
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
    "route_key",
    "_source",
    "_file_name",
    "_file_id",
)


class SqliteSparseIndex:
    """Maintain and search a local SQLite FTS5 sparse index."""

    def __init__(self, db_path: str | None = None) -> None:
        self.db_path = db_path or getattr(config, "rag_sparse_index_path", "data/rag_sparse.sqlite")
        self._initialize()

    def upsert_documents(self, documents: Sequence[Document]) -> None:
        if not documents:
            return

        # 一个 context 内的全部语句在退出时统一 commit，异常则不会执行显式 commit。
        with self._connection() as conn:
            for doc in documents:
                metadata = dict(doc.metadata or {})
                chunk_id = str(metadata.get("_chunk_id") or metadata.get("id") or "")
                if not chunk_id:
                    logger.warning("跳过缺少 _chunk_id 的稀疏索引文档")
                    continue
                row = self._row_from_document(chunk_id, doc, metadata)
                # FTS5 虚表使用“先删后插”实现确定性 upsert，主键由稳定 chunk_id 提供。
                conn.execute("DELETE FROM rag_sparse WHERE chunk_id = ?", (chunk_id,))
                conn.execute(
                    """
                    INSERT INTO rag_sparse (
                        chunk_id, content, metadata_text, metadata_json, file_id,
                        source, document_type, route_key
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    row,
                )

    def replace_file_documents(
        self,
        file_id: str,
        documents: Sequence[Document],
    ) -> int:
        """Atomically write one file generation, then remove its obsolete chunks."""
        if not documents:
            raise ValueError("replacement documents must not be empty")

        # 先完整校验所有 ID，再开启写事务，避免写到一半才发现某个 chunk 无身份。
        chunk_ids = [str((doc.metadata or {}).get("_chunk_id") or "") for doc in documents]
        if any(not chunk_id for chunk_id in chunk_ids):
            raise ValueError("replacement documents require _chunk_id")

        # 新 chunk 写入与旧 chunk 清理共用一个 SQLite 事务，读者不会看到半个 generation。
        with self._connection() as conn:
            for chunk_id, doc in zip(chunk_ids, documents, strict=True):
                metadata = dict(doc.metadata or {})
                row = self._row_from_document(chunk_id, doc, metadata)
                conn.execute("DELETE FROM rag_sparse WHERE chunk_id = ?", (chunk_id,))
                conn.execute(
                    """
                    INSERT INTO rag_sparse (
                        chunk_id, content, metadata_text, metadata_json, file_id,
                        source, document_type, route_key
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    row,
                )

            # 仅删除同 file_id 且不在新 ID 集合中的行，刚写入的 generation 被明确保留。
            placeholders = ", ".join("?" for _ in chunk_ids)
            cursor = conn.execute(
                f"DELETE FROM rag_sparse WHERE file_id = ? AND chunk_id NOT IN ({placeholders})",
                [file_id, *chunk_ids],
            )
            return int(cursor.rowcount or 0)

    def contains_file_hash(self, file_id: str, file_hash: str) -> bool:
        """Return whether the sparse index contains the requested file generation."""
        # file_hash 位于完整 JSON metadata 中；limit 1 足以证明 generation 已同步。
        with self._connection() as conn:
            row = conn.execute(
                """
                SELECT 1
                FROM rag_sparse
                WHERE file_id = ?
                  AND json_extract(metadata_json, '$._file_hash') = ?
                LIMIT 1
                """,
                (file_id, file_hash),
            ).fetchone()
        return row is not None

    def delete_by_file_id(self, file_id: str) -> int:
        with self._connection() as conn:
            cursor = conn.execute("DELETE FROM rag_sparse WHERE file_id = ?", (file_id,))
            return int(cursor.rowcount or 0)

    def delete_by_source(self, source: str) -> int:
        with self._connection() as conn:
            cursor = conn.execute("DELETE FROM rag_sparse WHERE source = ?", (source,))
            return int(cursor.rowcount or 0)

    def search(
        self,
        query: str,
        k: int = 10,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[tuple[LangChainDocument, float]]:
        # 先把用户查询转换成 FTS 精确 token 的 OR 表达式；无有效词时不访问数据库。
        terms = self._query_terms(query)
        if not terms:
            return []

        where_clauses = ["rag_sparse MATCH ?"]
        params: list[Any] = [terms]
        # 过滤条件全部参数化绑定；字段名只能来自固定白名单，值不能注入 SQL。
        for key, value in (metadata_filter or {}).items():
            if key not in METADATA_FIELDS or value is None or value == "":
                return []
            # 列表值表示同字段多值匹配，用 IN 展开；任一非法空值都会拒绝整个过滤查询。
            if isinstance(value, list):
                values = [str(item) for item in value if item is not None and item != ""]
                if not values or len(values) != len(value):
                    return []
                placeholders = ", ".join("?" for _ in values)
                where_clauses.append(f"json_extract(metadata_json, '$.{key}') IN ({placeholders})")
                params.extend(values)
                continue
            where_clauses.append(f"json_extract(metadata_json, '$.{key}') = ?")
            params.append(str(value))
        params.append(int(k))

        # bm25 越小通常越相关，ORDER BY 默认升序；返回时取绝对值供领域层统一保存。
        sql = f"""
            SELECT chunk_id, content, metadata_json, bm25(rag_sparse) AS score
            FROM rag_sparse
            WHERE {" AND ".join(where_clauses)}
            ORDER BY score
            LIMIT ?
        """
        try:
            with self._connection() as conn:
                rows = conn.execute(sql, params).fetchall()
        except sqlite3.OperationalError as exc:
            logger.warning(f"稀疏检索失败: {exc}")
            return []

        results: list[tuple[LangChainDocument, float]] = []
        # 读取出口恢复完整 metadata，并转换为 LangChain Document 供兼容 search API 使用。
        for row in rows:
            metadata = json.loads(row["metadata_json"] or "{}")
            metadata.setdefault("_chunk_id", row["chunk_id"])
            score = abs(float(row["score"] or 0.0))
            results.append(
                (LangChainDocument(page_content=row["content"], metadata=metadata), score)
            )
        return results

    def search_hits(
        self,
        query: str,
        k: int = 10,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[SearchHit]:
        """Return sparse candidates using framework-neutral models."""
        # 端口方法再次把兼容类型转为领域 SearchHit，source 明确标记为 sparse。
        return [
            SearchHit(
                document=Document(
                    page_content=document.page_content,
                    metadata=dict(document.metadata),
                    id=getattr(document, "id", None),
                ),
                score=float(score),
                source="sparse",
            )
            for document, score in self.search(query, k=k, metadata_filter=metadata_filter)
        ]

    def _initialize(self) -> None:
        path = Path(self.db_path)
        if path.parent and str(path.parent) != ".":
            path.parent.mkdir(parents=True, exist_ok=True)
        # metadata_json 用于过滤与还原但不参与全文索引；content/metadata_text 才参与 MATCH。
        with self._connection() as conn:
            conn.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS rag_sparse USING fts5(
                    chunk_id UNINDEXED,
                    content,
                    metadata_text,
                    metadata_json UNINDEXED,
                    file_id UNINDEXED,
                    source UNINDEXED,
                    document_type UNINDEXED,
                    route_key UNINDEXED,
                    tokenize = 'unicode61'
                )
                """)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    @contextmanager
    def _connection(self) -> Generator[sqlite3.Connection, None, None]:
        # contextmanager 集中管理 commit/close，调用方法不需要各自维护连接生命周期。
        conn = self._connect()
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _row_from_document(
        self,
        chunk_id: str,
        doc: Document,
        metadata: dict[str, Any],
    ) -> tuple[str, str, str, str, str, str, str, str]:
        # 把常用结构字段拼入可索引文本，让型号、告警名等即使不在正文也能被 FTS 命中。
        metadata_text = " ".join(
            str(metadata.get(key, "")) for key in METADATA_FIELDS if metadata.get(key)
        )
        return (
            chunk_id,
            doc.page_content,
            metadata_text,
            # sort_keys 产生稳定 JSON，ensure_ascii=False 保留中文以便调试与直接检查数据库。
            json.dumps(metadata, ensure_ascii=False, sort_keys=True),
            str(metadata.get("_file_id", "")),
            str(metadata.get("_source", "")),
            str(metadata.get("document_type", "")),
            str(metadata.get("route_key", "")),
        )

    def _query_terms(self, query: str) -> str:
        # 每个 token 用引号包裹并转义内部引号，避免 FTS 操作符改变原查询含义。
        terms = TOKEN_PATTERN.findall(query)
        escaped = [term.replace('"', '""') for term in terms if term.strip()]
        return " OR ".join(f'"{term}"' for term in escaped)
