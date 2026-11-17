"""Build immutable retrieval plans from analyzed and rewritten user queries."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from app.rag.models import RetrievalProfile
from app.rag.query.metadata_registry import MetadataFieldRegistry, default_metadata_registry
from app.rag.query.models import (
    QueryAnalysis,
    QueryPlan,
    QueryRequest,
    RewriteDecision,
    RewrittenQuery,
)

_MAX_SUBQUERIES = 8


def _positive_int(value: object, default: int) -> int:
    """Accept only real positive integers; bools and floats are configuration errors."""
    if isinstance(value, int) and not isinstance(value, bool) and value >= 1:
        return value
    return default


def _bounded_subqueries(value: object) -> int:
    """Keep fan-out deterministic even if settings came from untrusted configuration."""
    if type(value) is not int or value <= 0:
        return 0
    return min(value, _MAX_SUBQUERIES)


@dataclass(frozen=True, slots=True)
class QueryPlannerSettings:
    """Static limits and feature switches for declarative retrieval plans."""

    top_k: int = 3
    aiops_top_k: int = 8
    max_top_k: int = 100
    route_enabled: bool = True
    hybrid_enabled: bool = True
    hybrid_profiles: tuple[RetrievalProfile, ...] = (
        RetrievalProfile.AIOPS,
        RetrievalProfile.PROTOCOL,
    )
    rrf_k: int = 60
    route_top_k: int = 2
    sparse_top_k: int = 20
    max_subqueries: int = 2
    max_attempts: int = 8

    def __post_init__(self) -> None:
        if type(self.max_top_k) is not int or self.max_top_k < 1:
            raise ValueError("max_top_k must be a non-boolean positive integer")
        if type(self.route_enabled) is not bool:
            raise ValueError("route_enabled must be a boolean")
        if type(self.hybrid_enabled) is not bool:
            raise ValueError("hybrid_enabled must be a boolean")
        if not isinstance(self.hybrid_profiles, tuple) or not all(
            isinstance(profile, RetrievalProfile) for profile in self.hybrid_profiles
        ):
            raise ValueError("hybrid_profiles must be a tuple of RetrievalProfile values")

        object.__setattr__(self, "top_k", _clamp_top_k(self.top_k, 3, self.max_top_k))
        object.__setattr__(self, "aiops_top_k", _clamp_top_k(self.aiops_top_k, 8, self.max_top_k))
        object.__setattr__(self, "rrf_k", _positive_int(self.rrf_k, 60))
        depth_maximum = self.max_top_k * 3
        object.__setattr__(
            self, "route_top_k", _clamp_depth(self.route_top_k, 2, depth_maximum)
        )
        object.__setattr__(
            self, "sparse_top_k", _clamp_depth(self.sparse_top_k, 20, depth_maximum)
        )
        object.__setattr__(self, "max_subqueries", _bounded_subqueries(self.max_subqueries))
        object.__setattr__(self, "max_attempts", _positive_int(self.max_attempts, 1))


class DefaultQueryPlanner:
    """Convert query-stage values into one bounded, side-effect-free plan."""

    def __init__(
        self,
        settings: QueryPlannerSettings | None = None,
        metadata_registry: MetadataFieldRegistry | None = None,
    ) -> None:
        self._settings = settings or QueryPlannerSettings()
        self._metadata_registry = metadata_registry or default_metadata_registry()

    def plan(
        self,
        request: QueryRequest,
        analysis: QueryAnalysis,
        decision: RewriteDecision,
        rewrite: RewrittenQuery,
    ) -> QueryPlan:
        """Create a plan without mutating or retaining caller-owned values."""
        plan_request = deepcopy(request)
        plan_analysis = deepcopy(analysis)
        plan_decision = deepcopy(decision)
        plan_rewrite = deepcopy(rewrite)
        profile = _normalize_profile(plan_analysis.profile)
        request_profile = _normalize_profile(plan_request.profile)
        if request_profile is not profile:
            raise ValueError("request and analysis profile must match")
        final_top_k = self._final_top_k(plan_request.top_k, profile)
        metadata_filter = self._metadata_filter(plan_request, plan_analysis, plan_rewrite, profile)

        return QueryPlan(
            request=plan_request,
            analysis=plan_analysis,
            decision=plan_decision,
            rewrite=plan_rewrite,
            queries=self._queries(plan_rewrite),
            metadata_filter=metadata_filter,
            final_top_k=final_top_k,
            candidate_k=self._candidate_k(final_top_k, profile),
            route_first=self._settings.route_enabled and request.metadata_filter is None,
            hybrid=self._settings.hybrid_enabled and profile in self._settings.hybrid_profiles,
            rrf_k=_positive_int(self._settings.rrf_k, 60),
            max_attempts=_positive_int(self._settings.max_attempts, 1),
            route_k=self._settings.route_top_k,
            sparse_min_k=self._settings.sparse_top_k,
            explicit_filter=plan_request.metadata_filter is not None,
        )

    def _final_top_k(self, requested_top_k: object, profile: RetrievalProfile) -> int:
        if (
            isinstance(requested_top_k, int)
            and not isinstance(requested_top_k, bool)
            and requested_top_k >= 1
        ):
            return min(requested_top_k, self._settings.max_top_k)
        default = (
            self._settings.aiops_top_k
            if profile is RetrievalProfile.AIOPS
            else self._settings.top_k
        )
        return _clamp_top_k(default, 1, self._settings.max_top_k)

    def _candidate_k(self, final_top_k: int, profile: RetrievalProfile) -> int:
        if profile in (RetrievalProfile.AIOPS, RetrievalProfile.PROTOCOL):
            candidate_k = max(final_top_k * 3, 15)
        else:
            candidate_k = max(final_top_k * 2, 6)
        return min(candidate_k, self._settings.max_top_k * 3)

    def _queries(self, rewrite: RewrittenQuery) -> tuple[str, ...]:
        queries = [rewrite.primary_text] if rewrite.primary_text else []
        seen = set(queries)
        for query in rewrite.subqueries:
            if not query or query in seen:
                continue
            if len(queries) - (1 if rewrite.primary_text else 0) >= self._settings.max_subqueries:
                break
            seen.add(query)
            queries.append(query)
        if not queries:
            raise ValueError("query plan must contain at least one query")
        return tuple(queries)

    def _metadata_filter(
        self,
        request: QueryRequest,
        analysis: QueryAnalysis,
        rewrite: RewrittenQuery,
        profile: RetrievalProfile,
    ) -> dict[str, Any]:
        # An explicit filter is a caller boundary: preserve it exactly and never add guesses.
        if request.metadata_filter is not None:
            return deepcopy(request.metadata_filter)

        metadata_filter = deepcopy(analysis.inferred_filter)
        for candidate in rewrite.entity_candidates:
            normalized = self._metadata_registry.normalize_candidate(
                candidate,
                request.text,
                request.conversation_context,
                profile,
            )
            if normalized is not None:
                if normalized.field not in metadata_filter:
                    metadata_filter[normalized.field] = normalized.value
                elif metadata_filter[
                    normalized.field
                ] != normalized.value and self._metadata_registry.requires_strict_match(
                    {normalized.field: normalized.value}
                ):
                    raise ValueError(
                        f"conflicting strict metadata candidate for field {normalized.field!r}"
                    )
        return metadata_filter


def _normalize_profile(profile: object) -> RetrievalProfile:
    try:
        return RetrievalProfile(profile)
    except (TypeError, ValueError):
        return RetrievalProfile.CHAT


def _clamp_top_k(value: object, fallback: int, maximum: int) -> int:
    """Return a bounded positive count while rejecting bool and float impostors."""
    if not isinstance(value, int) or isinstance(value, bool):
        value = fallback
    return min(max(value, 1), maximum)


def _clamp_depth(value: object, fallback: int, maximum: int) -> int:
    if type(value) is not int or value < 1:
        value = fallback
    return min(value, maximum)
