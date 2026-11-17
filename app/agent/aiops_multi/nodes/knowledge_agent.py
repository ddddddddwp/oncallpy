"""Knowledge collection node for the multi-agent AIOps workflow."""

from __future__ import annotations

import asyncio
from typing import Any

from loguru import logger

from app.agent.aiops_multi.state import MultiAgentAIOpsState
from app.agent.aiops_multi.tool_registry import build_error, stringify_tool_result


async def knowledge_agent(state: MultiAgentAIOpsState) -> dict[str, Any]:
    """Retrieve knowledge-base evidence for the current alert or incident."""

    logger.info("=== Knowledge Agent: collect knowledge evidence ===")
    query = state.get("input", "")
    evidence: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    try:
        from app.tools import retrieve_knowledge

        if hasattr(retrieve_knowledge, "ainvoke"):
            result = await retrieve_knowledge.ainvoke({"query": query, "profile": "aiops"})
        elif hasattr(retrieve_knowledge, "invoke"):
            result = await asyncio.to_thread(
                retrieve_knowledge.invoke,
                {"query": query, "profile": "aiops"},
            )
        else:
            result = await asyncio.to_thread(retrieve_knowledge, query)

        content = stringify_tool_result(result)
        if content:
            evidence.append(
                {
                    "source": "knowledge_base",
                    "query": query,
                    "content": content,
                }
            )
    except Exception as exc:  # pragma: no cover - depends on vector-store runtime
        errors.append(build_error("knowledge_agent", exc, tool="retrieve_knowledge"))

    partial: dict[str, Any] = {
        "current_stage": "knowledge_collection",
        "knowledge_evidence": evidence,
        "past_steps": [
            (
                "knowledge_agent",
                f"检索到 {len(evidence)} 条知识库证据" if evidence else "未检索到知识库证据",
            )
        ],
    }
    if errors:
        partial["errors"] = errors
    return partial


__all__ = ["knowledge_agent"]
