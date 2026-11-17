"""Rule-based, deterministic analysis for incoming RAG queries."""

from __future__ import annotations

from typing import Any

from app.rag.models.search_query import RetrievalProfile
from app.rag.protocol_metadata import infer_protocol_metadata_from_text
from app.rag.query.metadata_registry import MetadataFieldRegistry, default_metadata_registry
from app.rag.query.models import QueryAnalysis, QueryRequest
from app.rag.rag_taxonomy import infer_taxonomy_from_text

RUNBOOK_INTENT_KEYWORDS = (
    ("troubleshooting", ("排查", "检查", "定位", "troubleshoot")),
    ("resolution", ("处理", "解决", "修复", "处置", "resolution")),
    ("verification", ("验证", "确认", "verification")),
)
CONTEXT_REFERENCES = ("它", "这个", "那个", "刚才", "上一个")


class DefaultQueryAnalyzer:
    """Infer safe metadata filters without an LLM or external state."""

    def __init__(self, registry: MetadataFieldRegistry | None = None) -> None:
        self._registry = registry or default_metadata_registry()

    @property
    def registry(self) -> MetadataFieldRegistry:
        """Expose the registry used to validate this analyzer's inferred filters."""
        return self._registry

    def analyze(self, request: QueryRequest) -> QueryAnalysis:
        normalized_text = " ".join(request.text.split())
        profile = _normalize_profile(request.profile)
        taxonomy_filter = infer_taxonomy_from_text(normalized_text)
        intents = _infer_runbook_intents(normalized_text) if profile == RetrievalProfile.AIOPS else ()
        inferred_filter: dict[str, Any] = dict(taxonomy_filter)

        if profile == RetrievalProfile.AIOPS:
            inferred_filter = {"document_type": "runbook", **inferred_filter}
            if intents:
                inferred_filter["runbook_section"] = (
                    intents[0] if len(intents) == 1 else list(intents)
                )
        elif profile == RetrievalProfile.PROTOCOL:
            inferred_filter = {
                "document_type": "protocol",
                **inferred_filter,
                **infer_protocol_metadata_from_text(request.text),
            }

        inferred_filter = self._registry.validate_filter(inferred_filter, profile)
        context_dependent = any(reference in normalized_text for reference in CONTEXT_REFERENCES)
        reasons: list[str] = []
        confidence = 0.9
        if context_dependent:
            confidence -= 0.35
            reasons.append("context_reference")
        if len(intents) > 1:
            confidence -= 0.2
            reasons.append("multiple_intents")
        if not inferred_filter and len(normalized_text) > 80:
            confidence -= 0.2
            reasons.append("long_unstructured_query")
        confidence = round(max(0.0, min(confidence, 1.0)), 10)

        return QueryAnalysis(
            normalized_text=normalized_text,
            profile=profile,
            intents=intents,
            entities=dict(taxonomy_filter),
            inferred_filter=inferred_filter,
            complexity=(
                "contextual"
                if context_dependent
                else "compound"
                if len(intents) > 1
                else "simple"
            ),
            confidence=confidence,
            context_dependent=context_dependent,
            reasons=tuple(reasons),
        )


def _normalize_profile(profile: RetrievalProfile | str) -> RetrievalProfile:
    try:
        return RetrievalProfile(profile)
    except ValueError:
        return RetrievalProfile.CHAT


def _infer_runbook_intents(text: str) -> tuple[str, ...]:
    text_lower = text.lower()
    return tuple(
        intent
        for intent, keywords in RUNBOOK_INTENT_KEYWORDS
        if any(keyword in text_lower for keyword in keywords)
    )
