"""Deterministic policy deciding whether a query needs LLM rewriting."""

from __future__ import annotations

from dataclasses import dataclass

from app.rag.query.models import QueryAnalysis, QueryRequest, RewriteDecision, RewriteMode


@dataclass(frozen=True, slots=True)
class RewritePolicySettings:
    """Configuration for the conservative automatic rewrite decision."""

    llm_enabled: bool
    confidence_threshold: float = 0.75


class DefaultRewritePolicy:
    """Choose an LLM only when automatic rewriting has a clear reason."""

    def __init__(self, settings: RewritePolicySettings) -> None:
        self._settings = settings

    def decide(self, request: QueryRequest, analysis: QueryAnalysis) -> RewriteDecision:
        mode = normalize_rewrite_mode(request.rewrite_mode)
        if mode in (RewriteMode.OFF, RewriteMode.RULE):
            return RewriteDecision(use_llm=False)
        if mode is RewriteMode.LLM:
            return RewriteDecision(
                use_llm=self._settings.llm_enabled,
                reasons=() if self._settings.llm_enabled else ("llm_disabled",),
            )
        if request.metadata_filter:
            return RewriteDecision(use_llm=False, reasons=("explicit_metadata_filter",))
        reasons = _auto_rewrite_reasons(analysis, self._settings.confidence_threshold)
        if not reasons:
            return RewriteDecision(use_llm=False)
        if not self._settings.llm_enabled:
            return RewriteDecision(use_llm=False, reasons=(*reasons, "llm_disabled"))
        return RewriteDecision(use_llm=True, reasons=reasons)


def normalize_rewrite_mode(value: RewriteMode | str) -> RewriteMode:
    """Treat malformed caller values as the safe no-rewrite setting."""
    try:
        return RewriteMode(value)
    except ValueError:
        return RewriteMode.OFF


def _auto_rewrite_reasons(
    analysis: QueryAnalysis, confidence_threshold: float
) -> tuple[str, ...]:
    reasons: list[str] = []
    if analysis.confidence < confidence_threshold:
        reasons.append("low_confidence")
    if len(analysis.intents) > 1:
        reasons.append("multiple_intents")
    if analysis.context_dependent:
        reasons.append("context_dependent")
    if analysis.complexity == "compound":
        reasons.append("compound_query")
    return tuple(reasons)
