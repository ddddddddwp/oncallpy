from datetime import datetime, timedelta

from app.rag.models import Document, IndexingResult, RetrievalProfile, SearchHit, SearchQuery


def test_document_metadata_is_not_shared() -> None:
    first = Document(page_content="one")
    second = Document(page_content="two")

    first.metadata["key"] = "value"

    assert second.metadata == {}


def test_search_query_and_hit_use_domain_documents() -> None:
    query = SearchQuery(text="CPUHigh", profile=RetrievalProfile.AIOPS, top_k=2)
    hit = SearchHit(document=Document("runbook"), score=0.8, source="dense")

    assert query.profile == RetrievalProfile.AIOPS
    assert hit.document.page_content == "runbook"


def test_indexing_result_preserves_legacy_dictionary() -> None:
    result = IndexingResult(directory_path="/tmp/docs", total_files=1)
    result.start_time = datetime(2026, 7, 22, 10, 0, 0)
    result.end_time = result.start_time + timedelta(milliseconds=25)
    result.increment_success_count()
    result.add_failed_file("bad.md", "broken")

    assert result.to_dict() == {
        "success": False,
        "directory_path": "/tmp/docs",
        "total_files": 1,
        "success_count": 1,
        "skipped_count": 0,
        "fail_count": 0,
        "deleted_count": 0,
        "inserted_chunk_count": 0,
        "duration_ms": 25,
        "error_message": "",
        "failed_files": {"bad.md": "broken"},
    }
