"""文档分割服务模块 - 基于 LangChain 的智能文档分割"""

# LangChain 类型只存在于本实现内部；公开结果会在出口转换为 app.rag.models.Document。
import hashlib
import re
from pathlib import Path

from langchain_core.documents import Document as LangChainDocument
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from loguru import logger

from app.config import config
from app.rag.models.document import Document
from app.rag.protocol_metadata import extract_markdown_protocol_table_rows


class DocumentSplitterService:
    """文档分割服务 - 使用 LangChain 的分割器"""

    def __init__(self):
        """初始化文档分割服务"""
        # chunk_size 是主配置；Markdown 二次切分使用两倍大小以减少标题切分后的碎片。
        self.chunk_size = config.chunk_max_size
        self.chunk_overlap = config.chunk_overlap

        # Markdown 标题分割器 (只按二级标题分割，避免一级/三级标题造成碎片化)
        self.markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("##", "h2"),
            ],
            strip_headers=False,  # 保留标题在内容中
        )

        # 递归字符分割器 (用于二次分割，使用更大的chunk_size)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size * 2,  # 加倍chunk_size，减少分片数
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )

        logger.info(
            f"文档分割服务初始化完成, chunk_size={self.chunk_size}, "
            f"secondary_chunk_size={self.chunk_size * 2}, "
            f"overlap={self.chunk_overlap}"
        )

    def split_markdown(self, content: str, file_path: str = "") -> list[Document]:
        """
        分割 Markdown 文档 (两阶段分割 + 合并小片段)

        Args:
            content: Markdown 内容
            file_path: 文件路径 (用于元数据)

        Returns:
            List[Document]: 文档分片列表
        """
        if not content or not content.strip():
            logger.warning(f"Markdown 文档内容为空: {file_path}")
            return []

        try:
            # 第一阶段: 按标题分割
            md_docs = self.markdown_splitter.split_text(content)

            # 第二阶段: 按大小进一步分割
            docs_after_split = self.text_splitter.split_documents(md_docs)

            # 第三阶段: 补充运维文档结构元数据
            self._enrich_runbook_metadata(docs_after_split)

            # 第四阶段: 合并太小的分片 (< 300字符)，但不跨 runbook 章节合并
            final_docs = self._merge_small_chunks(docs_after_split, min_size=300)

            # 添加文件路径元数据
            for doc in final_docs:
                doc.metadata["_source"] = file_path
                doc.metadata["_extension"] = ".md"
                doc.metadata["_file_name"] = Path(file_path).name

            # 协议表每一行另建精确 chunk，避免寄存器地址被大段 Markdown 稀释。
            for row_content, row_metadata in extract_markdown_protocol_table_rows(content):
                row_metadata["_source"] = file_path
                row_metadata["_extension"] = ".md"
                row_metadata["_file_name"] = Path(file_path).name
                final_docs.append(
                    LangChainDocument(page_content=row_content, metadata=row_metadata)
                )

            # sentence-window 用小单元参与召回，同时把邻近句保存为生成上下文。
            if config.rag_sentence_window_enabled:
                final_docs = self._to_sentence_window_documents(final_docs)
            else:
                self._assign_chunk_indexes(final_docs)

            logger.info(f"Markdown 分割完成: {file_path} -> {len(final_docs)} 个分片")
            return [self._to_domain_document(doc) for doc in final_docs]

        except Exception as e:
            logger.error(f"Markdown 分割失败: {file_path}, 错误: {e}")
            raise

    def split_text(self, content: str, file_path: str = "") -> list[Document]:
        """
        分割普通文本文档

        Args:
            content: 文本内容
            file_path: 文件路径 (用于元数据)

        Returns:
            List[Document]: 文档分片列表
        """
        if not content or not content.strip():
            logger.warning(f"文本文档内容为空: {file_path}")
            return []

        try:
            # 直接使用递归字符分割器
            docs = self.text_splitter.create_documents(
                texts=[content],
                metadatas=[
                    {
                        "_source": file_path,
                        "_extension": Path(file_path).suffix,
                        "_file_name": Path(file_path).name,
                    }
                ],
            )

            if config.rag_sentence_window_enabled:
                docs = self._to_sentence_window_documents(docs)
            else:
                self._assign_chunk_indexes(docs)

            logger.info(f"文本分割完成: {file_path} -> {len(docs)} 个分片")
            return [self._to_domain_document(doc) for doc in docs]

        except Exception as e:
            logger.error(f"文本分割失败: {file_path}, 错误: {e}")
            raise

    def split_document(self, content: str, file_path: str = "") -> list[Document]:
        """
        智能分割文档 (根据文件类型选择分割器)

        Args:
            content: 文档内容
            file_path: 文件路径

        Returns:
            List[Document]: 文档分片列表
        """
        # 只有 Markdown 需要标题/协议表增强，其他扩展名统一走纯文本递归切分。
        if file_path.endswith(".md"):
            return self.split_markdown(content, file_path)
        else:
            return self.split_text(content, file_path)

    def _merge_small_chunks(
        self,
        documents: list[LangChainDocument],
        min_size: int = 300,
    ) -> list[LangChainDocument]:
        """
        合并太小的分片

        Args:
            documents: 文档列表
            min_size: 最小分片大小 (字符数)

        Returns:
            List[Document]: 合并后的文档列表
        """
        if not documents:
            return []

        merged_docs: list[LangChainDocument] = []
        current_doc: LangChainDocument | None = None

        # current_doc 充当待提交缓冲区：只有确认下一个 chunk 不能合并时才写入结果。
        for doc in documents:
            doc_size = len(doc.page_content)

            if current_doc is None:
                # 第一个文档
                current_doc = doc
            # 合并同时受“小 chunk、总长度上限、结构边界”三项约束。
            elif (
                doc_size < min_size
                and len(current_doc.page_content) < self.chunk_size * 2
                and self._can_merge_chunks(current_doc, doc)
            ):
                # 当前文档太小且合并后不会太大，则合并
                current_doc.page_content += "\n\n" + doc.page_content
                # 保留主文档的元数据
            else:
                # 保存当前文档，开始新文档
                merged_docs.append(current_doc)
                current_doc = doc

        # 添加最后一个文档
        if current_doc is not None:
            merged_docs.append(current_doc)

        return merged_docs

    def _can_merge_chunks(
        self,
        left: LangChainDocument,
        right: LangChainDocument,
    ) -> bool:
        """判断两个 chunk 是否可以合并。"""
        # 不跨 runbook 语义章节合并，否则排查步骤与回滚/验证内容会被混成一个候选。
        left_section = left.metadata.get("runbook_section")
        right_section = right.metadata.get("runbook_section")
        if left_section and right_section and left_section != right_section:
            return False

        # 即使尚未推断 section，也用 h2 保住 Markdown 的主要结构边界。
        left_h2 = left.metadata.get("h2")
        right_h2 = right.metadata.get("h2")
        if left_h2 and right_h2 and left_h2 != right_h2:
            return False

        return True

    def _enrich_runbook_metadata(self, documents: list[LangChainDocument]) -> None:
        """为运维 Markdown 文档补充告警名和章节类型。"""
        # 先找文档级默认告警名；后续 chunk 若出现更具体名称，会更新向后传播的上下文。
        alert_name = self._find_first_alert_name(documents)

        for doc in documents:
            inferred_alert = self._extract_alert_name(doc.page_content)
            if inferred_alert:
                alert_name = inferred_alert

            if alert_name:
                doc.metadata.setdefault("alert_name", alert_name)

            section = self._infer_runbook_section(doc)
            doc.metadata["runbook_section"] = section

    def _find_first_alert_name(self, documents: list[LangChainDocument]) -> str:
        # 正文中的显式“告警名称”可信度最高，标题只作为找不到显式字段时的回退。
        for doc in documents:
            alert_name = self._extract_alert_name(doc.page_content)
            if alert_name:
                return alert_name
        for doc in documents:
            for key in ("h2", "h1"):
                heading = doc.metadata.get(key)
                if isinstance(heading, str) and heading.strip():
                    return heading.strip()
        return ""

    def _extract_alert_name(self, content: str) -> str:
        patterns = [
            r"告警名称\s*[:：]\s*([^\n\r]+)",
            r"告警名\s*[:：]\s*([^\n\r]+)",
            r"Alert\s*[:：]\s*([^\n\r]+)",
        ]
        for pattern in patterns:
            match = re.search(pattern, content, flags=re.IGNORECASE)
            if match:
                return match.group(1).strip().strip("-* ")
        return ""

    def _infer_runbook_section(self, doc: LangChainDocument) -> str:
        # 标题权重通过拼入 sample 自然体现；正文只取前 200 字，减少跨主题误判。
        heading = " ".join(
            str(doc.metadata.get(key, "")) for key in ("h1", "h2", "h3") if doc.metadata.get(key)
        )
        sample = f"{heading}\n{doc.page_content[:200]}".lower()

        # 判断顺序也是优先级：一个片段命中多个关键词时，首先匹配的类别胜出。
        if any(keyword in sample for keyword in ("排查", "检查", "定位", "troubleshoot")):
            return "troubleshooting"
        if any(keyword in sample for keyword in ("处理方案", "解决", "修复", "处置", "resolution")):
            return "resolution"
        if any(keyword in sample for keyword in ("回滚", "rollback")):
            return "rollback"
        if any(keyword in sample for keyword in ("验证", "确认", "verification")):
            return "verification"
        if any(keyword in sample for keyword in ("症状", "现象", "symptom")):
            return "symptom"
        if any(keyword in sample for keyword in ("告警", "alert")):
            return "alert"
        return "other"

    def _assign_chunk_indexes(self, documents: list[LangChainDocument]) -> None:
        # 索引在最终候选列表稳定后统一分配，保证 _chunk_count 与实际写入数量一致。
        chunk_count = len(documents)
        for index, doc in enumerate(documents):
            doc.metadata["_chunk_index"] = index
            doc.metadata["_chunk_count"] = chunk_count

    def _to_sentence_window_documents(
        self,
        documents: list[LangChainDocument],
    ) -> list[LangChainDocument]:
        """Convert base chunks into smaller indexed units with generation context."""
        # base chunk 是上下文边界；窗口永远不会跨越两个父 chunk。
        window_docs: list[LangChainDocument] = []
        window_size = max(0, config.rag_sentence_window_size)

        for parent_index, doc in enumerate(documents):
            # retrieval unit 是实际 embedding/FTS 的正文，父 chunk 只提供稳定身份和邻域。
            units = self._split_retrieval_units(doc.page_content)
            if not units:
                continue

            # parent_id 让查询阶段能识别同一父块的多个句子窗口并进行去重/追踪。
            parent_id = self._stable_parent_chunk_id(doc, parent_index)
            for unit_index, unit in enumerate(units):
                # 在列表边缘裁剪窗口，_window_* 字段保留实际覆盖的闭区间。
                start = max(0, unit_index - window_size)
                end = min(len(units), unit_index + window_size + 1)
                metadata = dict(doc.metadata)
                metadata.update(
                    {
                        "_original_text": unit,
                        "_context_window": " ".join(units[start:end]),
                        "_index_granularity": "sentence_window",
                        "_window_start_index": start,
                        "_window_end_index": end - 1,
                        "_parent_chunk_id": parent_id,
                        "_parent_chunk_index": parent_index,
                        "_sentence_index": unit_index,
                    }
                )
                window_docs.append(LangChainDocument(page_content=unit, metadata=metadata))

        # 所有文本都短于阈值时不能返回空索引，退回原始 chunk 粒度。
        if not window_docs:
            for doc in documents:
                doc.metadata["_index_granularity"] = "chunk"
            self._assign_chunk_indexes(documents)
            return documents

        self._assign_chunk_indexes(window_docs)
        return window_docs

    def _stable_parent_chunk_id(self, doc: LangChainDocument, parent_index: int) -> str:
        # 来源、父序号和完整正文共同参与 hash，内容变化会自然产生新父身份。
        source = str(doc.metadata.get("_source", ""))
        value = f"{source}:{parent_index}:{doc.page_content}"
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    def _split_retrieval_units(self, content: str) -> list[str]:
        # 先按行保留 Markdown/表格的天然边界，再按中英文句末符号细分。
        units: list[str] = []
        min_chars = max(1, config.rag_sentence_min_unit_chars)

        for line in content.splitlines():
            compact = re.sub(r"\s+", " ", line).strip()
            if not compact:
                continue

            # 正则把终止标点保留在句子中，embedding 仍能看到原始语气与格式线索。
            sentence_parts = re.findall(r"[^。！？!?\.]+[。！？!?\.]+|[^。！？!?\.]+$", compact)
            parts = [part.strip() for part in sentence_parts if part.strip()]
            if not parts:
                parts = [compact]

            # 过短单元通常是标题符号或噪声；阈值由配置决定而不是在算法中硬编码。
            units.extend(part for part in parts if len(part) >= min_chars)

        if units:
            return units

        compact = re.sub(r"\s+", " ", content).strip()
        if len(compact) >= min_chars:
            return [compact]
        return []

    def _to_domain_document(self, document: LangChainDocument) -> Document:
        # 在实现边界复制 metadata，避免领域对象与 LangChain 对象共享同一个可变字典。
        return Document(
            page_content=document.page_content,
            metadata=dict(document.metadata),
            id=getattr(document, "id", None),
        )


# 全局单例
# 兼容旧调用路径；新用例优先通过 bootstrap 取得服务。
document_splitter_service = DocumentSplitterService()
