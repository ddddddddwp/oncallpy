"""Metadata-filter inference and route mapping for RAG queries."""

from __future__ import annotations

from typing import Any

from app.rag.models.search_query import RetrievalProfile
from app.rag.query.analyzer import DefaultQueryAnalyzer
from app.rag.query.metadata_registry import MetadataFieldRegistry, default_metadata_registry
from app.rag.query.models import QueryRequest


class DefaultQueryRouter:
    """Preserve the existing profile, taxonomy, and route-filter rules."""

    def __init__(
        self,
        analyzer: DefaultQueryAnalyzer | None = None,
        registry: MetadataFieldRegistry | None = None,
    ) -> None:
        if analyzer is not None:
            if registry is not None and registry is not analyzer.registry:
                raise ValueError("registry must be analyzer.registry when both are provided")
            self._analyzer = analyzer
            self._registry = registry or analyzer.registry
        else:
            self._registry = registry or default_metadata_registry()
            self._analyzer = DefaultQueryAnalyzer(self._registry)

    def infer_metadata_filter(
        self,
        query: str,
        profile: RetrievalProfile,
    ) -> dict[str, Any]:
        return self._analyzer.analyze(
            QueryRequest(text=query, profile=profile)
        ).inferred_filter

    def metadata_filter_fallbacks(
        self,
        metadata_filter: dict[str, Any],
    ) -> list[dict[str, Any]]:
        return self._registry.fallbacks(metadata_filter)

    def route_summary_filter(
        self,
        query_filter: dict[str, Any] | None,
    ) -> dict[str, Any]:
        result: dict[str, Any] = {"document_type": "route_summary"}
        for key in (
            "device_type",
            "manufacturer",
            "model",
            "protocol_type",
            "register_address",
            "function_code",
        ):
            if query_filter and query_filter.get(key):
                result[key] = query_filter[key]
        return result

    def requires_taxonomy_match(self, metadata_filter: dict[str, Any]) -> bool:
        return any(
            metadata_filter.get(key)
            for key in (
                "device_type",
                "manufacturer",
                "model",
                "register_address",
                "function_code",
            )
        )

    def filter_from_route_metadata(
        self,
        metadata: dict[str, Any],
        profile: RetrievalProfile,
    ) -> dict[str, Any]:
        target_type = metadata.get("target_document_type") or self.default_document_type(profile)
        result: dict[str, Any] = {"document_type": target_type}
        for key in (
            "alert_name",
            "business_domain",
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
        ):
            if metadata.get(key):
                result[key] = metadata[key]
        return result

    def default_document_type(self, profile: RetrievalProfile) -> str:
        if profile == RetrievalProfile.PROTOCOL:
            return "protocol"
        if profile == RetrievalProfile.AIOPS:
            return "runbook"
        return "document"
