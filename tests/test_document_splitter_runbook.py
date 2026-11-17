from app.rag.document_splitter_service import DocumentSplitterService
from app.rag.indexing.document_splitter import (
    DocumentSplitterService as NewDocumentSplitterService,
)
from app.rag.models import Document


def test_legacy_splitter_reexports_domain_splitter() -> None:
    assert DocumentSplitterService is NewDocumentSplitterService


def test_markdown_splitter_preserves_runbook_structure():
    content = """
# CPU 使用率过高

## 告警详情
- 告警名称: CPUHigh
- 服务: checkout

## 排查步骤
1. 查看 CPU 指标
2. 查询错误日志

## 处理方案
扩容 checkout 实例。
""".strip()

    docs = DocumentSplitterService().split_markdown(content, "/tmp/cpu.md")

    assert docs
    assert all(isinstance(doc, Document) for doc in docs)
    assert all("_chunk_index" in doc.metadata for doc in docs)
    assert all("_chunk_count" in doc.metadata for doc in docs)
    assert any(doc.metadata.get("alert_name") == "CPUHigh" for doc in docs)
    assert any(doc.metadata.get("runbook_section") == "troubleshooting" for doc in docs)
    assert any(doc.metadata.get("runbook_section") == "resolution" for doc in docs)


def test_markdown_splitter_uses_only_h2_as_section_boundary(monkeypatch):
    monkeypatch.setattr(
        "app.rag.document_splitter_service.config.rag_sentence_window_enabled",
        False,
    )
    content = """
# 一级标题 A
alpha

# 一级标题 B
beta

### 三级标题 C
gamma
""".strip()

    docs = DocumentSplitterService().split_markdown(content, "/tmp/general.md")

    assert len(docs) == 1
    assert "一级标题 A" in docs[0].page_content
    assert "一级标题 B" in docs[0].page_content
    assert "三级标题 C" in docs[0].page_content


def test_markdown_splitter_does_not_emit_sentence_windows_by_default():
    content = """
# CPU 使用率过高

## 告警详情
- 告警名称: CPUHigh

## 排查步骤
先查看 CPU 指标。再查询 checkout 错误日志。最后确认是否存在突增流量。
""".strip()

    service = DocumentSplitterService()
    docs = service.split_markdown(content, "/tmp/cpu.md")

    assert docs
    assert all(doc.metadata.get("_index_granularity") != "sentence_window" for doc in docs)
    assert any("再查询 checkout 错误日志。" in doc.page_content for doc in docs)


def test_markdown_splitter_emits_protocol_table_rows():
    content = """
# 英威腾逆变器 Modbus 寄存器协议

## 遥测寄存器

| 地址 | 名称 | 功能码 | 数据类型 | 单位 | 倍率 | 读写 |
| --- | --- | --- | --- | --- | --- | --- |
| 0x3601 | 告警状态 | 03 | uint16 | - | 1 | R |
| 0x3602 | 直流电压 | 03 | uint16 | V | 0.1 | R |
""".strip()

    docs = DocumentSplitterService().split_markdown(
        content,
        "/uploads/英威腾逆变器Modbus寄存器协议.md",
    )

    protocol_docs = [
        doc for doc in docs if doc.metadata.get("document_type") == "protocol"
    ]
    assert len(protocol_docs) == 2
    assert protocol_docs[0].metadata["register_address"] == "0x3601"
    assert protocol_docs[0].metadata["register_name"] == "告警状态"
    assert protocol_docs[0].metadata["function_code"] == "03"
    assert protocol_docs[0].metadata["data_type"] == "uint16"
    assert protocol_docs[0].metadata["read_write"] == "R"
    assert "0x3601" in protocol_docs[0].page_content
