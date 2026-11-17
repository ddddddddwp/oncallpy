"""Tool discovery and safe invocation helpers for multi-agent AIOps."""

from __future__ import annotations

import asyncio
import inspect
import json
from typing import Any


METRIC_TOOL_HINTS = (
    "metric",
    "metrics",
    "monitor",
    "prometheus",
    "grafana",
    "cpu",
    "memory",
    "mem",
    "disk",
    "latency",
    "qps",
    "throughput",
    "traffic",
)
LOG_TOOL_HINTS = (
    "log",
    "logs",
    "cls",
    "sls",
    "error",
    "exception",
    "trace",
)
QUERY_ARG_HINTS = ("query", "input", "question", "prompt", "content", "message", "text")


def build_error(stage: str, error: BaseException | str, **context: Any) -> dict[str, Any]:
    """Build a serializable error entry for workflow state."""

    message = str(error)
    entry: dict[str, Any] = {"stage": stage, "message": message}
    if context:
        entry.update(context)
    return entry


def get_tool_name(tool: Any) -> str:
    """Return a best-effort display name for a LangChain or MCP tool."""

    return str(getattr(tool, "name", None) or getattr(tool, "__name__", None) or type(tool).__name__)


def get_tool_description(tool: Any) -> str:
    """Return a best-effort description for a LangChain or MCP tool."""

    return str(getattr(tool, "description", "") or "")


def get_local_tools() -> tuple[list[Any], list[dict[str, Any]]]:
    """Load local tools lazily so package import stays side-effect light."""

    tools: list[Any] = []
    errors: list[dict[str, Any]] = []

    try:
        from app.tools.time_tool import get_current_time

        tools.append(get_current_time)
    except Exception as exc:  # pragma: no cover - depends on optional runtime setup
        errors.append(build_error("local_tool_load", exc, tool="get_current_time"))

    try:
        from app.tools.knowledge_tool import retrieve_knowledge

        tools.append(retrieve_knowledge)
    except Exception as exc:  # pragma: no cover - depends on optional Milvus runtime
        errors.append(build_error("local_tool_load", exc, tool="retrieve_knowledge"))

    return tools, errors


async def get_mcp_tools() -> tuple[list[Any], list[dict[str, Any]]]:
    """Load MCP tools lazily and convert connection failures into state errors."""

    try:
        from app.agent.mcp_client import get_mcp_client_with_retry

        mcp_client = await get_mcp_client_with_retry()
        tools = await mcp_client.get_tools()
        return list(tools or []), []
    except Exception as exc:  # pragma: no cover - requires MCP runtime
        return [], [build_error("mcp_tool_load", exc)]


async def get_available_tools(include_mcp: bool = True) -> tuple[list[Any], list[dict[str, Any]]]:
    """Return local tools plus MCP tools, preserving load failures as errors."""

    local_tools, errors = get_local_tools()
    if not include_mcp:
        return local_tools, errors

    mcp_tools, mcp_errors = await get_mcp_tools()
    return local_tools + mcp_tools, errors + mcp_errors


def format_tool_descriptions(tools: list[Any]) -> str:
    """Format tools using the existing AIOps utility with a local fallback."""

    try:
        from app.agent.aiops.utils import format_tools_description

        description = format_tools_description(tools)
        if description:
            return description
    except Exception:
        pass

    lines = []
    for tool in tools:
        description = get_tool_description(tool)
        if description:
            lines.append(f"- {get_tool_name(tool)}: {description}")
        else:
            lines.append(f"- {get_tool_name(tool)}")
    return "\n".join(lines)


def classify_tool(tool: Any) -> str:
    """Roughly classify an ops tool as metrics, logs, or other by name/description."""

    haystack = f"{get_tool_name(tool)} {get_tool_description(tool)}".lower()
    if any(hint in haystack for hint in METRIC_TOOL_HINTS):
        return "metrics"
    if any(hint in haystack for hint in LOG_TOOL_HINTS):
        return "logs"
    return "other"


def select_ops_tools(tools: list[Any], max_per_category: int = 3) -> list[tuple[str, Any]]:
    """Select a small set of metric/log tools to avoid broad MCP fan-out."""

    selected: list[tuple[str, Any]] = []
    counts = {"metrics": 0, "logs": 0}

    for tool in tools:
        category = classify_tool(tool)
        if category not in counts:
            continue
        if counts[category] >= max_per_category:
            continue
        selected.append((category, tool))
        counts[category] += 1

    return selected


def build_tool_payload(tool: Any, query: str) -> Any:
    """Create a conservative payload for a single-input tool."""

    args = getattr(tool, "args", None)
    if isinstance(args, dict) and args:
        for key in QUERY_ARG_HINTS:
            if key in args:
                return {key: query}
        if len(args) == 1:
            return {next(iter(args)): query}

    return {"query": query}


async def invoke_tool(tool: Any, query: str) -> tuple[str, dict[str, Any] | None]:
    """Invoke a tool without allowing exceptions to escape."""

    payload = build_tool_payload(tool, query)
    try:
        if hasattr(tool, "ainvoke"):
            result = await tool.ainvoke(payload)
        elif hasattr(tool, "invoke"):
            result = await asyncio.to_thread(tool.invoke, payload)
        elif callable(tool):
            result = await _call_callable_tool(tool, payload)
        else:
            raise TypeError(f"Tool {get_tool_name(tool)} is not invokable")
    except Exception as exc:  # pragma: no cover - runtime integration dependent
        return "", build_error(
            "tool_invoke",
            exc,
            tool=get_tool_name(tool),
            category=classify_tool(tool),
        )

    return stringify_tool_result(result), None


async def _call_callable_tool(tool: Any, payload: Any) -> Any:
    if isinstance(payload, dict):
        try:
            result = tool(**payload)
        except TypeError:
            result = tool(payload)
    else:
        result = tool(payload)

    if inspect.isawaitable(result):
        return await result
    return result


def stringify_tool_result(result: Any) -> str:
    """Convert common tool and message return shapes into text."""

    if isinstance(result, tuple) and result:
        result = result[0]

    content = getattr(result, "content", None)
    if content is not None:
        return str(content)

    if isinstance(result, str):
        return result

    if isinstance(result, (dict, list)):
        return json.dumps(result, ensure_ascii=False, default=str)

    return str(result)


__all__ = [
    "build_error",
    "classify_tool",
    "format_tool_descriptions",
    "get_available_tools",
    "get_local_tools",
    "get_mcp_tools",
    "get_tool_name",
    "invoke_tool",
    "select_ops_tools",
    "stringify_tool_result",
]
