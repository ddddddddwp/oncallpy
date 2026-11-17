"""Metadata-field rules shared by query analysis, planning, and retrieval."""

from __future__ import annotations

import re
import unicodedata
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from app.rag.models.search_query import RetrievalProfile
from app.rag.protocol_metadata import infer_protocol_metadata_from_text
from app.rag.query.models import EntityCandidate, QueryTurn
from app.rag.rag_taxonomy import infer_taxonomy_from_text


@dataclass(frozen=True, slots=True)
class MetadataFieldSpec:
    """Describe whether one canonical metadata field may be used as a filter."""

    name: str
    aliases: tuple[str, ...] = ()
    filterable: bool = True
    relaxable: bool = False
    strict: bool = False
    profiles: tuple[RetrievalProfile, ...] = ()


DEFAULT_FIELD_SPECS = (
    MetadataFieldSpec("document_type"),
    MetadataFieldSpec("runbook_section"),
    MetadataFieldSpec("alert_name", strict=True),
    MetadataFieldSpec("business_domain"),
    MetadataFieldSpec("device_type", strict=True),
    MetadataFieldSpec("manufacturer", relaxable=True),
    MetadataFieldSpec("model", strict=True),
    MetadataFieldSpec("protocol_type"),
    MetadataFieldSpec("register_address", strict=True),
    MetadataFieldSpec("register_name"),
    MetadataFieldSpec("function_code", strict=True),
    MetadataFieldSpec("data_type"),
    MetadataFieldSpec("unit"),
    MetadataFieldSpec("scale"),
    MetadataFieldSpec("read_write"),
    MetadataFieldSpec("route_key"),
    MetadataFieldSpec("_source"),
)

MAX_CANDIDATE_FIELD_CHARS = 80
MAX_CANDIDATE_VALUE_CHARS = 160
MAX_CANDIDATE_EVIDENCE_CHARS = 500


class MetadataFieldRegistry:
    """Centralize allowed metadata filters and safe fallback decisions."""

    def __init__(self, field_specs: tuple[MetadataFieldSpec, ...] = DEFAULT_FIELD_SPECS) -> None:
        if len({spec.name for spec in field_specs}) != len(field_specs):
            raise ValueError("metadata field names must be unique")
        self._by_name = {spec.name: spec for spec in field_specs}
        self._aliases: dict[str, str] = {}
        for spec in field_specs:
            for alias in (spec.name, *spec.aliases):
                existing = self._aliases.get(alias)
                if existing and existing != spec.name:
                    raise ValueError(f"metadata alias {alias!r} maps to multiple fields")
                self._aliases[alias] = spec.name

    def validate_filter(
        self,
        metadata_filter: dict[str, Any],
        profile: RetrievalProfile | str | None = None,
    ) -> dict[str, Any]:
        """Return the non-empty, filterable subset with aliases canonicalized."""
        normalized_profile = _normalize_profile(profile)
        validated: dict[str, Any] = {}
        for canonical, value in self._canonicalize_filter(metadata_filter).items():
            spec = self._by_name.get(canonical)
            if not spec or not spec.filterable or not _has_value(value):
                continue
            if spec.profiles and normalized_profile not in spec.profiles:
                continue
            validated[spec.name] = deepcopy(value)
        return validated

    def validate_candidate(
        self,
        candidate: EntityCandidate,
        text: str,
        conversation_context: tuple[QueryTurn, ...] = (),
        profile: RetrievalProfile | str | None = None,
    ) -> bool:
        """Whether the candidate can be normalized into a safe metadata value."""
        return self.normalize_candidate(candidate, text, conversation_context, profile) is not None

    def normalize_candidate(
        self,
        candidate: EntityCandidate,
        text: str,
        conversation_context: tuple[QueryTurn, ...] = (),
        profile: RetrievalProfile | str | None = None,
    ) -> EntityCandidate | None:
        """Canonicalize only candidates whose value is bound to their own evidence."""
        field = _normalized_text(candidate.field)
        value = _normalized_text(candidate.value)
        evidence = _normalized_text(candidate.evidence)
        if (
            not field
            or not value
            or not evidence
            or len(field) > MAX_CANDIDATE_FIELD_CHARS
            or len(value) > MAX_CANDIDATE_VALUE_CHARS
            or len(evidence) > MAX_CANDIDATE_EVIDENCE_CHARS
        ):
            return None
        canonical = self._aliases.get(field)
        spec = self._by_name.get(canonical or "")
        if (
            canonical is None
            or not spec
            or not spec.filterable
            or (spec.profiles and _normalize_profile(profile) not in spec.profiles)
        ):
            return None
        sources = (text, *(turn.content for turn in conversation_context))
        if not any(contains_bounded_text(source, evidence) for source in sources):
            return None
        canonical_value = _canonical_candidate_value(canonical, value, evidence)
        if not canonical_value:
            return None
        return EntityCandidate(field=canonical, value=canonical_value, evidence=evidence)

    def fallbacks(self, metadata_filter: dict[str, Any]) -> list[dict[str, Any]]:
        """Generate exact-first fallbacks by dropping one allowed relaxable field."""
        original = self._canonicalize_filter(metadata_filter)
        candidates = [deepcopy(original)]
        for field, value in original.items():
            spec = self._by_name.get(field)
            if spec and spec.relaxable and not spec.strict and _has_value(value):
                relaxed = deepcopy(original)
                relaxed.pop(field)
                candidates.append(relaxed)

        unique: list[dict[str, Any]] = []
        seen: set[tuple[tuple[str, str], ...]] = set()
        for candidate in candidates:
            signature = tuple(sorted((key, str(value)) for key, value in candidate.items()))
            if signature not in seen:
                seen.add(signature)
                unique.append(candidate)
        return unique

    def requires_strict_match(self, metadata_filter: dict[str, Any]) -> bool:
        """Whether a filter contains an entity that must never be relaxed."""
        canonical_filter = self._canonicalize_filter(metadata_filter)
        return any(
            spec.strict and _has_value(canonical_filter.get(name))
            for name, spec in self._by_name.items()
        )

    def _canonicalize_filter(self, metadata_filter: dict[str, Any]) -> dict[str, Any]:
        """Copy a filter and reject ambiguous canonical-name/alias duplicates."""
        canonical_filter: dict[str, Any] = {}
        for field, value in metadata_filter.items():
            canonical = self._aliases.get(field, field)
            if canonical in canonical_filter:
                raise ValueError(f"multiple values supplied for metadata field {canonical!r}")
            canonical_filter[canonical] = deepcopy(value)
        return canonical_filter

def default_metadata_registry() -> MetadataFieldRegistry:
    """Build the default registry without sharing mutable state between callers."""
    return MetadataFieldRegistry()


def _normalize_profile(profile: RetrievalProfile | str | None) -> RetrievalProfile:
    try:
        return RetrievalProfile(profile or RetrievalProfile.CHAT)
    except ValueError:
        return RetrievalProfile.CHAT


def _has_value(value: Any) -> bool:
    if isinstance(value, str):
        return bool(value.strip())
    return bool(value)


def contains_bounded_text(source: str, value: str) -> bool:
    """Match Unicode text safely and short ASCII identifiers only on boundaries."""
    normalized_source = _normalized_text(source).casefold()
    normalized_value = _normalized_text(value).casefold()
    if not normalized_value:
        return False
    left_boundary = r"(?<![a-z0-9])" if normalized_value[0].isascii() and normalized_value[0].isalnum() else ""
    right_boundary = r"(?![a-z0-9])" if normalized_value[-1].isascii() and normalized_value[-1].isalnum() else ""
    return bool(re.search(rf"{left_boundary}{re.escape(normalized_value)}{right_boundary}", normalized_source))


def normalized_text_equal(left: str, right: str) -> bool:
    return unicodedata.normalize("NFKC", left).casefold().strip() == unicodedata.normalize(
        "NFKC", right
    ).casefold().strip()


def _canonical_candidate_value(canonical_field: str, value: str, evidence: str) -> str:
    inferred_evidence = {
        **infer_taxonomy_from_text(evidence),
        **infer_protocol_metadata_from_text(evidence),
    }
    inferred_value = inferred_evidence.get(canonical_field)
    inferred_candidate_fields = {
        **infer_taxonomy_from_text(value),
        **infer_protocol_metadata_from_text(value),
    }
    inferred_candidate = inferred_candidate_fields.get(canonical_field, value)
    if inferred_candidate_fields and canonical_field not in inferred_candidate_fields:
        return ""
    if inferred_value:
        if normalized_text_equal(inferred_value, inferred_candidate):
            return inferred_value
        return ""
    return value if contains_bounded_text(evidence, value) else ""


def _normalized_text(value: str) -> str:
    return unicodedata.normalize("NFKC", value).strip()
