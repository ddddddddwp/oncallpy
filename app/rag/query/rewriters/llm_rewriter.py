"""Bounded validation around LLM-produced query rewrites."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from app.rag.models import RetrievalProfile
from app.rag.protocol_metadata import infer_protocol_metadata_from_text
from app.rag.query.contracts import QueryRewriteModel
from app.rag.query.metadata_registry import (
    MetadataFieldRegistry,
    contains_bounded_text,
    default_metadata_registry,
    normalized_text_equal,
)
from app.rag.query.models import EntityCandidate, QueryAnalysis, QueryRequest, RewrittenQuery
from app.rag.rag_taxonomy import infer_taxonomy_from_text


class QueryRewriteValidationError(ValueError):
    """The untrusted rewrite-model response cannot safely be used for retrieval."""


MAX_REQUEST_TEXT_CHARS = 8_000
MAX_NORMALIZED_TEXT_CHARS = 8_000
MAX_INTENTS = 8
MAX_INTENT_CHARS = 80
MAX_ENTITIES = 16
MAX_ENTITY_KEY_CHARS = 80
MAX_ENTITY_VALUE_CHARS = 160
MAX_ENTITY_CANDIDATES = 16
MAX_REASON_CHARS = 500
MAX_ALLOWED_SUBQUERIES = 16


class LLMQueryRewriter:
    """Call a rewrite model with bounded context and validate its whole response."""

    def __init__(
        self,
        model: QueryRewriteModel,
        max_subqueries: int = 3,
        registry: MetadataFieldRegistry | None = None,
    ) -> None:
        if (
            not isinstance(max_subqueries, int)
            or isinstance(max_subqueries, bool)
            or not 0 <= max_subqueries <= MAX_ALLOWED_SUBQUERIES
        ):
            raise ValueError(f"max_subqueries must be an integer from 0 to {MAX_ALLOWED_SUBQUERIES}")
        self._model = model
        self.max_subqueries = max_subqueries
        self._registry = registry or default_metadata_registry()

    def rewrite(self, request: QueryRequest, analysis: QueryAnalysis) -> RewrittenQuery:
        payload = self._payload(request, analysis)
        output = self._model.generate(payload)
        if not isinstance(output, dict):
            raise QueryRewriteValidationError("model output must be a dictionary")

        primary_text = _validated_query(output.get("primary_text"), "primary_text")
        subqueries = _validated_subqueries(output.get("subqueries", ()), self.max_subqueries)
        candidates = self._validated_candidates(
            output.get("entity_candidates", ()), request, analysis.profile
        )
        preserved_terms = _protected_terms(analysis)
        searchable_text = "\n".join((primary_text, *subqueries))
        for field, term in _protected_entities(analysis):
            if not _preserves_entity(searchable_text, field, term):
                raise QueryRewriteValidationError(f"protected term missing: {term}")

        confidence = output.get("confidence", 1.0)
        if (
            not isinstance(confidence, int | float)
            or isinstance(confidence, bool)
            or not math.isfinite(confidence)
            or not 0.0 <= confidence <= 1.0
        ):
            raise QueryRewriteValidationError("confidence must be a finite number from 0 to 1")
        reason = output.get("reason", "")
        if not isinstance(reason, str) or len(reason) > MAX_REASON_CHARS:
            raise QueryRewriteValidationError("reason must be a bounded string")
        return RewrittenQuery(
            primary_text=primary_text,
            subqueries=subqueries,
            preserved_terms=preserved_terms,
            entity_candidates=candidates,
            source="llm",
            confidence=float(confidence),
            reason=reason,
        )

    def _payload(self, request: QueryRequest, analysis: QueryAnalysis) -> dict[str, Any]:
        _validate_payload_input(request, analysis)
        return {
            "text": request.text,
            "profile": analysis.profile.value,
            "normalized_text": analysis.normalized_text,
            "intents": list(analysis.intents),
            "entities": dict(analysis.entities),
            "conversation_context": [
                {"role": turn.role, "content": turn.content[-1000:]}
                for turn in request.conversation_context[-6:]
            ],
            "max_subqueries": self.max_subqueries,
        }

    def _validated_candidates(
        self,
        raw_candidates: object,
        request: QueryRequest,
        request_profile: RetrievalProfile | str,
    ) -> tuple[EntityCandidate, ...]:
        if not isinstance(raw_candidates, list | tuple):
            raise QueryRewriteValidationError("entity_candidates must be a list or tuple")
        if len(raw_candidates) > MAX_ENTITY_CANDIDATES:
            raise QueryRewriteValidationError("entity_candidates exceed maximum")
        candidates: list[EntityCandidate] = []
        for raw_candidate in raw_candidates:
            candidate = _to_entity_candidate(raw_candidate)
            normalized = (
                self._registry.normalize_candidate(
                    candidate, request.text, request.conversation_context, request_profile
                )
                if candidate
                else None
            )
            if not normalized:
                raise QueryRewriteValidationError("invalid entity candidate")
            candidates.append(normalized)
        return tuple(candidates)


def _validated_query(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise QueryRewriteValidationError(f"{name} must be a non-empty string")
    if len(value) > 500:
        raise QueryRewriteValidationError(f"{name} must not exceed 500 characters")
    return value


def _validated_subqueries(value: object, maximum: int) -> tuple[str, ...]:
    if not isinstance(value, list | tuple):
        raise QueryRewriteValidationError("subqueries must be a list or tuple")
    raw_subqueries = value
    if len(raw_subqueries) > maximum:
        raise QueryRewriteValidationError("subqueries exceed max_subqueries")
    unique: list[str] = []
    for index, subquery in enumerate(raw_subqueries):
        normalized = _validated_query(subquery, f"subqueries[{index}]")
        if normalized not in unique:
            unique.append(normalized)
    return tuple(unique)


def _to_entity_candidate(value: object) -> EntityCandidate | None:
    if not isinstance(value, Mapping):
        return None
    field = value.get("field")
    candidate_value = value.get("value")
    evidence = value.get("evidence")
    if (
        not isinstance(field, str)
        or not isinstance(candidate_value, str)
        or not isinstance(evidence, str)
    ):
        return None
    return EntityCandidate(field=field, value=candidate_value, evidence=evidence)


def _protected_terms(analysis: QueryAnalysis) -> tuple[str, ...]:
    return tuple(sorted({value for value in analysis.entities.values() if value}))


def _protected_entities(analysis: QueryAnalysis) -> tuple[tuple[str, str], ...]:
    return tuple(sorted((field, value) for field, value in analysis.entities.items() if value))


def _validate_payload_input(request: QueryRequest, analysis: QueryAnalysis) -> None:
    _bounded_string(request.text, "request.text", MAX_REQUEST_TEXT_CHARS)
    _bounded_string(analysis.normalized_text, "analysis.normalized_text", MAX_NORMALIZED_TEXT_CHARS)
    if len(analysis.intents) > MAX_INTENTS:
        raise QueryRewriteValidationError("analysis.intents exceed maximum")
    for intent in analysis.intents:
        _bounded_string(intent, "analysis.intents item", MAX_INTENT_CHARS)
    if len(analysis.entities) > MAX_ENTITIES:
        raise QueryRewriteValidationError("analysis.entities exceed maximum")
    for key, value in analysis.entities.items():
        _bounded_string(key, "analysis.entities key", MAX_ENTITY_KEY_CHARS)
        _bounded_string(value, "analysis.entities value", MAX_ENTITY_VALUE_CHARS)


def _bounded_string(value: object, name: str, maximum: int) -> None:
    if not isinstance(value, str) or len(value) > maximum:
        raise QueryRewriteValidationError(f"{name} must be a string no longer than {maximum}")


def _preserves_entity(rewritten_text: str, field: str, value: str) -> bool:
    if contains_bounded_text(rewritten_text, value):
        return True
    inferred = {
        **infer_taxonomy_from_text(rewritten_text),
        **infer_protocol_metadata_from_text(rewritten_text),
    }
    inferred_value = inferred.get(field)
    return bool(inferred_value) and normalized_text_equal(inferred_value, value)
