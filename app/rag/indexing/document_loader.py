"""Document loading utilities for the RAG ingestion pipeline."""

# loader 只负责建立可靠的文件事实，不承担切分、taxonomy 推断或存储写入。
from __future__ import annotations

import hashlib
from pathlib import Path

from app.rag.indexing.contracts import LoadedDocument


class RagDocumentLoader:
    """Load supported text documents and compute stable identity metadata."""

    def load(self, file_path: str) -> LoadedDocument:
        # resolve 将相对路径、.. 等表示统一成稳定绝对路径，避免同一文件产生多个身份。
        path = Path(file_path).resolve()
        if not path.exists() or not path.is_file():
            raise ValueError(f"文件不存在: {file_path}")

        # 内容 hash 基于磁盘原始字节，而不是解析后的正文；frontmatter 变化也会形成新 generation。
        raw_bytes = path.read_bytes()
        raw_text = raw_bytes.decode("utf-8")
        # frontmatter 与正文分开保存，结构字段可作为初始 metadata 参与后续标准化。
        metadata, text = self._parse_frontmatter(raw_text)
        stat = path.stat()
        normalized_path = path.as_posix()

        return LoadedDocument(
            path=str(path),
            normalized_path=normalized_path,
            file_name=path.name,
            extension=path.suffix.lower(),
            file_size=stat.st_size,
            modified_time=stat.st_mtime,
            raw_bytes=raw_bytes,
            text=text,
            # 路径 hash 稳定追踪“哪个文件”，字节 hash 稳定追踪“哪一版内容”。
            file_id=self._sha256_text(normalized_path),
            file_hash=self._sha256_bytes(raw_bytes),
            metadata=metadata,
        )

    def _parse_frontmatter(self, text: str) -> tuple[dict[str, str], str]:
        # 只识别文件开头成对的 ---；普通正文中的分隔线不会被误当成元数据。
        lines = text.splitlines()
        if not lines or lines[0].strip() != "---":
            return {}, text

        # next(..., None) 在结束标记缺失时安全退回原文，避免吞掉不完整文档。
        end_index = next(
            (index for index, line in enumerate(lines[1:], start=1) if line.strip() == "---"),
            None,
        )
        if end_index is None:
            return {}, text

        metadata: dict[str, str] = {}
        # 这是有意保持轻量的 key:value 解析，不执行 YAML 标签或复杂类型构造。
        for line in lines[1:end_index]:
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or ":" not in stripped:
                continue
            # 只在第一个冒号切分，因此 URL 或描述中的后续冒号仍属于 value。
            key, value = stripped.split(":", 1)
            key = key.strip()
            value = value.strip().strip("'\"")
            if key and value:
                metadata[key] = value

        body = "\n".join(lines[end_index + 1 :])
        return metadata, body

    def _sha256_text(self, value: str) -> str:
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    def _sha256_bytes(self, value: bytes) -> str:
        return hashlib.sha256(value).hexdigest()
