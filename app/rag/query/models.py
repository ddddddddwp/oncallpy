"""Domain values exchanged by the independent query pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Literal

from app.rag.models import Document, RetrievalProfile, SearchHit, SearchQuery


class RewriteMode(StrEnum):
    """Available query-rewrite strategies."""

    OFF = "off"
    RULE = "rule"
    AUTO = "auto"
    LLM = "llm"


@dataclass(frozen=True, slots=True)
class QueryTurn:
    role: Literal["user", "assistant"]
    content: str


@dataclass(frozen=True, slots=True)
class QueryRequest:
    text: str
    profile: RetrievalProfile | str = RetrievalProfile.CHAT
    top_k: int | None = None
    metadata_filter: dict[str, Any] | None = None
    conversation_context: tuple[QueryTurn, ...] = ()
    rewrite_mode: RewriteMode | str = RewriteMode.AUTO
    debug: bool = False

    @classmethod
    def from_search_query(cls, query: SearchQuery) -> QueryRequest:
        """Adapt the public RAG request without sharing its explicit filter."""
        return cls(
            text=query.text,
            profile=query.profile,
            top_k=query.top_k,
            metadata_filter=(
                dict(query.metadata_filter) if query.metadata_filter is not None else None
            ),
        )


@dataclass(frozen=True, slots=True)
class EntityCandidate:
    field: str
    value: str
    evidence: str


@dataclass(frozen=True, slots=True)
class QueryAnalysis:
    normalized_text: str
    profile: RetrievalProfile
    intents: tuple[str, ...] = ()
    entities: dict[str, str] = field(default_factory=dict)
    inferred_filter: dict[str, Any] = field(default_factory=dict)
    complexity: Literal["simple", "compound", "contextual"] = "simple"
    confidence: float = 1.0
    context_dependent: bool = False
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class RewriteDecision:
    use_llm: bool
    reasons: tuple[str, ...] = ()
    fallback: Literal["rule"] = "rule"


@dataclass(frozen=True, slots=True)
class RewrittenQuery:
    primary_text: str
    subqueries: tuple[str, ...] = ()
    preserved_terms: tuple[str, ...] = ()
    entity_candidates: tuple[EntityCandidate, ...] = ()
    source: Literal["off", "rule", "llm", "fallback"] = "rule"
    confidence: float = 1.0
    reason: str = ""


@dataclass(frozen=True, slots=True)
class QueryPlan:
    request: QueryRequest
    analysis: QueryAnalysis
    decision: RewriteDecision
    rewrite: RewrittenQuery
    queries: tuple[str, ...]
    metadata_filter: dict[str, Any] | None
    final_top_k: int
    candidate_k: int
    route_first: bool
    hybrid: bool
    rrf_k: int
    max_attempts: int
    # Kept defaulted for hand-built plans used by adapters and tests; the
    # planner always writes explicit bounded values from its settings.
    route_k: int = 2
    sparse_min_k: int = 20
    # Routing switches and caller-owned filters are separate concerns: a
    # deployment may disable routing while inferred filters remain relaxable.
    explicit_filter: bool = False


@dataclass(frozen=True, slots=True)
class QueryTraceEvent:
    stage: str
    status: Literal["started", "completed", "failed", "degraded"]
    duration_ms: float = 0.0
    data: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class QueryExecution:
    candidates: list[SearchHit] = field(default_factory=list)
    degraded: bool = False
    degradation_reasons: list[str] = field(default_factory=list)
    attempts_used: int = 0
    error: str | None = None
    route_attempted: bool = False
    route_matched: bool = False
    metadata_filter: dict[str, Any] | None = None


@dataclass(slots=True)
class QueryResult:
    documents: list[Document]
    plan: QueryPlan
    trace: tuple[QueryTraceEvent, ...] = ()
    degraded: bool = False
    degradation_reasons: tuple[str, ...] = ()
    error: str | None = None
