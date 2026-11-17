"""State definitions for the multi-agent AIOps workflow."""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict


class MultiAgentAIOpsState(TypedDict, total=False):
    """Shared state passed between multi-agent AIOps nodes."""

    input: str
    user_input: str
    session_id: str
    plan: list[str]
    past_steps: Annotated[list[tuple[str, str]], operator.add]
    current_stage: str

    tool_descriptions: str
    knowledge_evidence: Annotated[list[dict[str, Any]], operator.add]
    ops_evidence: Annotated[list[dict[str, Any]], operator.add]
    metrics_evidence: Annotated[list[dict[str, Any]], operator.add]
    log_evidence: Annotated[list[dict[str, Any]], operator.add]
    tool_results: Annotated[list[dict[str, Any]], operator.add]

    diagnosis: dict[str, Any]
    recommendations: list[dict[str, Any]]
    report: str
    response: str
    errors: Annotated[list[dict[str, Any]], operator.add]


__all__ = ["MultiAgentAIOpsState"]
