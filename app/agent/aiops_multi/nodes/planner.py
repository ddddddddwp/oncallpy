"""Planner node for the multi-agent AIOps workflow."""

from __future__ import annotations

from typing import Any

from loguru import logger
from pydantic import BaseModel, Field

from app.agent.aiops_multi.state import MultiAgentAIOpsState
from app.core.llm_factory import llm_factory
from app.agent.aiops_multi.tool_registry import (
    build_error,
    format_tool_descriptions,
    get_available_tools,
)


DEFAULT_PLAN = [
    "knowledge_agent: 检索知识库经验和历史处置方案",
    "ops_evidence_agent: 收集监控指标和日志证据",
    "diagnosis_agent: 汇总证据并形成诊断结论",
    "report_agent: 生成告警分析报告",
]


class Plan(BaseModel):
    """Structured planner output."""

    steps: list[str] = Field(description="按顺序执行的 AIOps 分析步骤")


async def planner(state: MultiAgentAIOpsState) -> dict[str, Any]:
    """Create an execution plan without letting integration failures escape."""

    logger.info("=== MultiAgent Planner: create plan ===")
    input_text = state.get("input", "")

    tools, errors = await get_available_tools(include_mcp=True)
    tools_description = format_tool_descriptions(tools)

    plan_steps = DEFAULT_PLAN.copy()
    llm_error = await _try_llm_plan(input_text, tools_description)
    if isinstance(llm_error, list):
        plan_steps = llm_error
    elif llm_error is not None:
        errors.append(llm_error)

    partial: dict[str, Any] = {
        "current_stage": "planning",
        "plan": plan_steps,
        "tool_descriptions": tools_description,
    }
    if errors:
        partial["errors"] = errors

    return partial


async def _try_llm_plan(
    input_text: str,
    tools_description: str,
) -> list[str] | dict[str, Any] | None:
    """Return plan steps on success, an error entry on failure, or None when skipped."""

    if not input_text:
        return None

    try:
        from langchain_core.messages import HumanMessage, SystemMessage
        llm = llm_factory.create_chat_model(temperature=0)
        structured_llm = llm.with_structured_output(Plan)
        result = await structured_llm.ainvoke(
            [
                SystemMessage(
                    content=(
                        "你是 AIOps 多智能体工作流规划器。"
                        "只输出适合 knowledge_agent、ops_evidence_agent、"
                        "diagnosis_agent、report_agent 顺序协作的简洁步骤。"
                        f"\n\n可用工具:\n{tools_description or '无'}"
                    )
                ),
                HumanMessage(content=f"需要分析的告警或问题:\n{input_text}"),
            ]
        )

        steps = result.steps if isinstance(result, Plan) else result.get("steps", [])
        steps = [str(step).strip() for step in steps if str(step).strip()]
        return steps or DEFAULT_PLAN.copy()
    except Exception as exc:  # pragma: no cover - requires live LLM runtime
        return build_error("planner_llm", exc)


__all__ = ["planner"]
