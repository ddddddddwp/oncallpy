import math

import pytest
from langchain_core.documents import Document

from app.rag.models import Document as DomainDocument, SearchHit
from app.rag.query.retrieval_fusion import WeightedQueryFusion
from app.rag.retrieval_fusion import reciprocal_rank_fusion


def test_rrf_merges_dense_and_sparse_results_by_chunk_id():
    shared = Document(page_content="CPUHigh precise runbook", metadata={"_chunk_id": "shared"})
    dense_only = Document(page_content="semantic cpu runbook", metadata={"_chunk_id": "dense"})
    sparse_only = Document(page_content="CPUHigh exact alert", metadata={"_chunk_id": "sparse"})

    fused = reciprocal_rank_fusion(
        dense_results=[(dense_only, 0.1), (shared, 0.2)],
        sparse_results=[(shared, 3.0), (sparse_only, 2.0)],
        k=60,
    )

    assert [doc.metadata["_chunk_id"] for doc, _score in fused] == ["shared", "dense", "sparse"]
    assert fused[0][0] is shared
    assert fused[0][0].metadata["_retrieval_sources"] == ["dense", "sparse"]
    assert fused[1][0].metadata["_retrieval_sources"] == ["dense"]
    assert fused[2][0].metadata["_retrieval_sources"] == ["sparse"]


def test_weighted_fusion_preserves_primary_priority_and_isolates_metadata():
    primary = SearchHit(DomainDocument("primary", {"_chunk_id": "primary"}), 0.1, "dense")
    secondary = SearchHit(DomainDocument("secondary", {"_chunk_id": "secondary"}), 0.1, "dense")

    fused = WeightedQueryFusion().fuse_ranked([(1.0, [primary]), (0.6, [secondary, primary])])

    assert [item.document.page_content for item in fused] == ["primary", "secondary"]
    assert fused[0].source == "fused"
    assert fused[0].document is not primary.document
    assert fused[0].document.metadata["_retrieval_score_kind"] == "relevance"
    fused[0].document.metadata["changed"] = True
    assert "changed" not in primary.document.metadata


def test_weighted_fusion_ignores_invalid_weights_and_rejects_invalid_k():
    hit = SearchHit(DomainDocument("one", {"_chunk_id": "one"}), 0.1, "dense")

    assert WeightedQueryFusion().fuse_ranked(
        [
            (0.0, [hit]),
            (-1.0, [hit]),
            (float("nan"), [hit]),
            (float("inf"), [hit]),
            (float("-inf"), [hit]),
        ]
    ) == []
    with pytest.raises(ValueError, match="positive"):
        WeightedQueryFusion().fuse_ranked([(1.0, [hit])], k=True)  # type: ignore[arg-type]


def test_weighted_fusion_scores_merges_sources_and_keeps_stable_ties():
    first = SearchHit(DomainDocument("first", {"_chunk_id": "same", "nested": {"a": []}}), 0.0, "dense")
    duplicate = SearchHit(DomainDocument("other copy", {"_chunk_id": "same"}), 0.0, "sparse")
    tied = SearchHit(DomainDocument("tied", {"id": "tied"}), 0.0, "dense")
    fallback_key = SearchHit(DomainDocument("fallback", {"_source": "manual"}), 0.0, "dense")

    fused = WeightedQueryFusion().fuse_ranked([(1.0, [first, tied, fallback_key]), (0.6, [duplicate])], k=10)

    assert fused[0].score == pytest.approx(1 / 11 + 0.6 / 11)
    assert fused[0].document.metadata["_retrieval_sources"] == ["dense", "sparse"]
    assert [item.document.page_content for item in fused[1:]] == ["tied", "fallback"]
    assert fused[2].document.metadata["_source"] == "manual"
    fused[0].document.metadata["nested"]["a"].append("changed")
    assert first.document.metadata["nested"] == {"a": []}

    stable = WeightedQueryFusion().fuse_ranked([(1.0, [tied]), (1.0, [fallback_key])], k=10)
    assert [item.document.page_content for item in stable] == ["tied", "fallback"]


@pytest.mark.parametrize("k", [0, -1, True, 1.5])
def test_weighted_fusion_rejects_every_invalid_k(k: object):
    hit = SearchHit(DomainDocument("one", {"_chunk_id": "one"}), 0.1, "dense")

    with pytest.raises(ValueError, match="positive"):
        WeightedQueryFusion().fuse_ranked([(1.0, [hit])], k=k)  # type: ignore[arg-type]


def test_weighted_fusion_counts_a_duplicate_document_once_per_query_list():
    a = SearchHit(DomainDocument("A", {"_chunk_id": "a"}), 0.0, "dense")
    b = SearchHit(DomainDocument("B", {"_chunk_id": "b"}), 0.0, "dense")

    fused = WeightedQueryFusion().fuse_ranked([(1.0, [b, a, a])], k=10)

    assert [item.document.page_content for item in fused] == ["B", "A"]
    assert fused[1].score == pytest.approx(1 / 12)


def test_weighted_fusion_uses_document_id_before_text_fallback():
    first = SearchHit(DomainDocument("same prefix content", {}, id="one"), 0.0, "dense")
    second = SearchHit(DomainDocument("same prefix content", {}, id="two"), 0.0, "dense")

    fused = WeightedQueryFusion().fuse_ranked([(1.0, [first, second])])

    assert [item.document.id for item in fused] == ["one", "two"]


def test_weighted_fusion_hashes_full_text_for_no_id_document_identity():
    prefix = "same long prefix " * 20
    first = SearchHit(DomainDocument(prefix + "first", {"_source": "manual"}), 0.0, "dense")
    second = SearchHit(DomainDocument(prefix + "second", {"_source": "manual"}), 0.0, "dense")
    duplicate = SearchHit(DomainDocument(prefix + "first", {"_source": "manual"}), 0.0, "sparse")

    fused = WeightedQueryFusion().fuse_ranked([(1.0, [first, second]), (0.6, [duplicate])])

    assert [item.document.page_content for item in fused] == [prefix + "first", prefix + "second"]


def test_weighted_fusion_rejects_impossibly_large_k_and_saturates_finite_scores():
    hit = SearchHit(DomainDocument("one", {"_chunk_id": "one"}), 0.1, "dense")

    with pytest.raises(ValueError, match="maximum"):
        WeightedQueryFusion().fuse_ranked([(1.0, [hit])], k=10**400)
    fused = WeightedQueryFusion().fuse_ranked([(1e308, [hit])] * 10, k=1)
    assert math.isfinite(fused[0].score)
