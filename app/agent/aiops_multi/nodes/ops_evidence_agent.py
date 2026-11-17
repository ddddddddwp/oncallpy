"""Operational evidence collection node for the multi-agent AIOps workflow."""

from __future__ import annotations

from typing import Any

from loguru import logger

from app.agent.aiops_multi.state import MultiAgentAIOpsState
from app.agent.aiops_multi.tool_registry import (
    build_error,
    format_tool_descriptions,
    get_mcp_tools,
    get_tool_name,
    invoke_tool,
    select_ops_tools,
)


async def ops_evidence_agent(state: MultiAgentAIOpsState) -> dict[str, Any]:
    """Collect metric and log evidence from MCP tools using rough name classification."""

    logger.info("=== Ops Evidence Agent: collect metrics/log evidence ===")
    query = state.get("input", "")
    errors: list[dict[str, Any]] = []
    metrics_evidence: list[dict[str, Any]] = []
    log_evidence: list[dict[str, Any]] = []
    tool_results: list[dict[str, Any]] = []

    observed_at = _safe_current_time(errors)
    mcp_tools, tool_errors = await get_mcp_tools()
    errors.extend(tool_errors)

    for category, tool in select_ops_tools(mcp_tools):
        content, error = await invoke_tool(tool, query)
        if error:
            errors.append(error)
            continue

        item = {
            "source": "mcp",
            "tool": get_tool_name(tool),
            "category": category,
            "query": query,
            "observed_at": observed_at,
            "content": content,
        }
        tool_results.append(item)
        if category == "metrics":
            metrics_evidence.append(item)
        elif category == "logs":
            log_evidence.append(item)

    ops_evidence = metrics_evidence + log_evidence
    partial: dict[str, Any] = {
        "current_stage": "ops_evidence_collection",
        "tool_descriptions": format_tool_descriptions(mcp_tools),
        "metrics_evidence": metrics_evidence,
        "log_evidence": log_evidence,
        "ops_evidence": ops_evidence,
        "tool_results": tool_results,
        "past_steps": [
            (
                "ops_evidence_agent",
                f"收集到 {len(metrics_evidence)} 条指标证据和 {len(log_evidence)} 条日志证据",
            )
        ],
    }
    if errors:
        partial["errors"] = errors
    return partial


def _safe_current_time(errors: list[dict[str, Any]]) -> str:
    try:
        from app.tools import get_current_time

        if hasattr(get_current_time, "invoke"):
            return str(get_current_time.invoke({"timezone": "Asia/Shanghai"}))
        return str(get_current_time("Asia/Shanghai"))
    except Exception as exc:  # pragma: no cover - local tool import/runtime dependent
        errors.append(build_error("ops_evidence_agent", exc, tool="get_current_time"))
        return ""


__all__ = ["ops_evidence_agent"]
