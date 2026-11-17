"""Report generation node for the multi-agent AIOps workflow."""

from __future__ import annotations

import json
from typing import Any

from loguru import logger

from app.agent.aiops_multi.state import MultiAgentAIOpsState
from app.core.llm_factory import llm_factory
from app.agent.aiops_multi.tool_registry import build_error, stringify_tool_result


async def report_agent(state: MultiAgentAIOpsState) -> dict[str, Any]:
    """Generate the final markdown report."""

    logger.info("=== Report Agent: generate report ===")
    errors: list[dict[str, Any]] = []
    report = _fallback_report(state)

    llm_result = await _try_llm_report(state)
    if isinstance(llm_result, dict) and "error" in llm_result:
        errors.append(llm_result["error"])
    elif isinstance(llm_result, str) and llm_result.strip():
        report = _ensure_report_heading(llm_result.strip())

    partial: dict[str, Any] = {
        "current_stage": "reporting",
        "report": report,
        "response": report,
        "past_steps": [("report_agent", "最终 Markdown 诊断报告已生成")],
    }
    if errors:
        partial["errors"] = errors
    return partial


async def _try_llm_report(state: MultiAgentAIOpsState) -> str | dict[str, Any] | None:
    try:
        from langchain_core.messages import HumanMessage, SystemMessage
        llm = llm_factory.create_chat_model(temperature=0)
        response = await llm.ainvoke(
            [
                SystemMessage(
                    content=(
                        "你是 AIOps 报告专家。输出 Markdown 报告，标题必须是"
                        "# 告警分析报告，内容包含问题概述、关键证据、诊断结论、建议动作。"
                    )
                ),
                HumanMessage(content=_report_context(state)),
            ]
        )
        return stringify_tool_result(response)
    except Exception as exc:  # pragma: no cover - requires live LLM runtime
        return {"error": build_error("report_llm", exc)}


def _fallback_report(state: MultiAgentAIOpsState) -> str:
    diagnosis = state.get("diagnosis", {})
    summary = diagnosis.get("summary", "暂无诊断结论") if isinstance(diagnosis, dict) else str(diagnosis)
    recommendations = state.get("recommendations", [])
    if not recommendations and isinstance(diagnosis, dict):
        recommendations = diagnosis.get("recommendations", [])

    lines = [
        "# 告警分析报告",
        "",
        "## 问题概述",
        state.get("input", "未提供告警输入。"),
        "",
        "## 关键证据",
        f"- 知识库证据: {len(state.get('knowledge_evidence', []))} 条",
        f"- 指标证据: {len(state.get('metrics_evidence', []))} 条",
        f"- 日志证据: {len(state.get('log_evidence', []))} 条",
        "",
        "## 诊断结论",
        summary,
        "",
        "## 建议动作",
    ]

    if recommendations:
        for item in recommendations:
            if isinstance(item, dict):
                lines.append(f"- {item.get('text') or item.get('action') or item}")
            else:
                lines.append(f"- {item}")
    else:
        lines.append("- 补充监控指标、日志和知识库证据后继续分析。")

    return "\n".join(lines)


def _ensure_report_heading(content: str) -> str:
    if content.startswith("# 告警分析报告"):
        return content
    return f"# 告警分析报告\n\n{content}"


def _report_context(state: MultiAgentAIOpsState) -> str:
    payload = {
        "input": state.get("input", ""),
        "diagnosis": state.get("diagnosis", {}),
        "knowledge_evidence": state.get("knowledge_evidence", []),
        "metrics_evidence": state.get("metrics_evidence", []),
        "log_evidence": state.get("log_evidence", []),
        "ops_evidence": state.get("ops_evidence", []),
        "previous_errors": state.get("errors", []),
    }
    return json.dumps(payload, ensure_ascii=False, default=str)


__all__ = ["report_agent"]
