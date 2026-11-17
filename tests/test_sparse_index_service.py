from langchain_core.documents import Document

from app.rag.infrastructure.sqlite_sparse_index import SqliteSparseIndex
from app.rag.models import Document as DomainDocument
from app.rag.sparse_index_service import SparseIndexService


def test_legacy_sparse_class_uses_infrastructure_implementation() -> None:
    assert issubclass(SparseIndexService, SqliteSparseIndex)


def test_sparse_index_searches_exact_professional_terms_with_metadata_filter(tmp_path):
    index_path = tmp_path / "rag_sparse.sqlite"
    service = SparseIndexService(db_path=str(index_path))
    docs = [
        Document(
            page_content="0x3601 告警状态 功能码 03 uint16",
            metadata={
                "_chunk_id": "protocol-1",
                "_source": "/uploads/invt.md",
                "document_type": "protocol",
                "manufacturer": "invt",
                "register_address": "0x3601",
            },
        ),
        Document(
            page_content="CPUHigh 排查步骤 查询 CPU 指标",
            metadata={
                "_chunk_id": "runbook-1",
                "_source": "/uploads/cpu.md",
                "document_type": "runbook",
                "alert_name": "CPUHigh",
            },
        ),
    ]

    service.upsert_documents(docs)

    results = service.search(
        "英威腾 0x3601 功能码 03",
        k=3,
        metadata_filter={"document_type": "protocol", "manufacturer": "invt"},
    )

    assert len(results) == 1
    assert results[0][0].metadata["_chunk_id"] == "protocol-1"
    assert results[0][0].metadata["register_address"] == "0x3601"
    assert results[0][1] > 0

    hits = service.search_hits(
        "英威腾 0x3601 功能码 03",
        k=3,
        metadata_filter={"document_type": "protocol", "manufacturer": "invt"},
    )
    assert len(hits) == 1
    assert hits[0].source == "sparse"
    assert isinstance(hits[0].document, DomainDocument)
    assert hits[0].document.metadata["_chunk_id"] == "protocol-1"


def test_sparse_index_deletes_documents_by_file_id(tmp_path):
    service = SparseIndexService(db_path=str(tmp_path / "rag_sparse.sqlite"))
    service.upsert_documents(
        [
            Document(
                page_content="CPUHigh 排查步骤",
                metadata={"_chunk_id": "chunk-1", "_file_id": "file-a"},
            )
        ]
    )

    assert service.search("CPUHigh", k=1)

    deleted = service.delete_by_file_id("file-a")

    assert deleted == 1
    assert service.search("CPUHigh", k=1) == []


def test_sparse_index_replaces_file_generation_and_reports_current_hash(tmp_path):
    service = SparseIndexService(db_path=str(tmp_path / "rag_sparse.sqlite"))
    service.upsert_documents(
        [
            Document(
                page_content="obsolete generation marker",
                metadata={
                    "_chunk_id": "old-chunk",
                    "_file_id": "file-a",
                    "_file_hash": "old-hash",
                },
            )
        ]
    )
    replacement = [
        Document(
            page_content="current generation marker",
            metadata={
                "_chunk_id": "new-chunk",
                "_file_id": "file-a",
                "_file_hash": "new-hash",
            },
        )
    ]

    deleted = service.replace_file_documents("file-a", replacement)

    assert deleted == 1
    assert service.search("obsolete", k=1) == []
    assert service.search("current", k=1)[0][0].metadata["_chunk_id"] == "new-chunk"
    assert service.contains_file_hash("file-a", "new-hash") is True
    assert service.contains_file_hash("file-a", "old-hash") is False


def _runbook_sparse_index(tmp_path):
    service = SparseIndexService(db_path=str(tmp_path / "rag_sparse.sqlite"))
    service.upsert_documents(
        [
            Document(
                page_content="CPUHigh 排查步骤",
                metadata={
                    "_chunk_id": "runbook-1",
                    "document_type": "runbook",
                    "runbook_section": "troubleshooting",
                },
            )
        ]
    )
    return service


def test_sparse_search_unknown_filter_key_fails_closed(tmp_path):
    service = _runbook_sparse_index(tmp_path)

    results = service.search("CPUHigh", metadata_filter={"unknown_key": "value"})

    assert results == []


def test_sparse_search_filters_runbook_section(tmp_path):
    service = _runbook_sparse_index(tmp_path)

    results = service.search(
        "CPUHigh",
        metadata_filter={"runbook_section": "resolution"},
    )

    assert results == []


def test_sparse_search_mixed_valid_and_invalid_filters_fail_closed(tmp_path):
    service = _runbook_sparse_index(tmp_path)

    results = service.search(
        "CPUHigh",
        metadata_filter={"document_type": "runbook", "unknown_key": "value"},
    )

    assert results == []
