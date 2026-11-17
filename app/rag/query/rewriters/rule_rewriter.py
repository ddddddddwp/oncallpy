"""Safe, deterministic query rewriting without external model calls."""

from __future__ import annotations

from app.rag.query.models import QueryAnalysis, QueryRequest, RewriteMode, RewrittenQuery
from app.rag.query.rewrite_policy import normalize_rewrite_mode


class RuleQueryRewriter:
    """Use analyzer-normalized text while retaining recognized entity values."""

    def rewrite(self, request: QueryRequest, analysis: QueryAnalysis) -> RewrittenQuery:
        protected_terms = tuple(sorted({value for value in analysis.entities.values() if value}))
        return RewrittenQuery(
            primary_text=analysis.normalized_text,
            preserved_terms=protected_terms,
            source="off" if normalize_rewrite_mode(request.rewrite_mode) is RewriteMode.OFF else "rule",
            confidence=analysis.confidence,
            reason="normalized_query",
        )
