"""Diagnosis node for the multi-agent AIOps workflow."""

from __future__ import annotations

import json
from typing import Any

from loguru import logger

from app.agent.aiops_multi.state import MultiAgentAIOpsState
from app.core.llm_factory import llm_factory
from app.agent.aiops_multi.tool_registry import build_error, stringify_tool_result


async def diagnosis_agent(state: MultiAgentAIOpsState) -> dict[str, Any]:
    """Analyze collected evidence and produce a diagnosis object."""

    logger.info("=== Diagnosis Agent: analyze evidence ===")
    errors: list[dict[str, Any]] = []
    diagnosis = _fallback_diagnosis(state)

    llm_result = await _try_llm_diagnosis(state)
    if isinstance(llm_result, dict) and "error" in llm_result:
        errors.append(llm_result["error"])
    elif isinstance(llm_result, str) and llm_result.strip():
        diagnosis = {
            "summary": llm_result.strip(),
            "confidence": "medium",
            "source": "llm",
        }

    partial: dict[str, Any] = {
        "current_stage": "diagnosis",
        "diagnosis": diagnosis,
        "recommendations": _extract_recommendations(diagnosis),
        "past_steps": [
            (
                "diagnosis_agent",
                str(diagnosis.get("summary", "诊断结论已生成"))
                if isinstance(diagnosis, dict)
                else "诊断结论已生成",
            )
        ],
    }
    if errors:
        partial["errors"] = errors
    return partial


def _extract_recommendations(diagnosis: dict[str, Any]) -> list[dict[str, Any]]:
    raw_recommendations = diagnosis.get("recommendations", [])
    if not isinstance(raw_recommendations, list):
        return [{"text": str(raw_recommendations)}]

    recommendations: list[dict[str, Any]] = []
    for item in raw_recommendations:
        if isinstance(item, dict):
            recommendations.append(item)
        else:
            recommendations.append({"text": str(item)})
    return recommendations


async def _try_llm_diagnosis(state: MultiAgentAIOpsState) -> str | dict[str, Any] | None:
    try:
        from langchain_core.messages import HumanMessage, SystemMessage
        llm = llm_factory.create_chat_model(temperature=0)
        response = await llm.ainvoke(
            [
                SystemMessage(
                    content=(
                        "你是 AIOps 诊断专家。基于知识库、指标、日志证据给出"
                        "根因假设、置信度、建议动作。不要编造缺失证据。"
                    )
                ),
                HumanMessage(content=_diagnosis_context(state)),
            ]
        )
        return stringify_tool_result(response)
    except Exception as exc:  # pragma: no cover - requires live LLM runtime
        return {"error": build_error("diagnosis_llm", exc)}


def _fallback_diagnosis(state: MultiAgentAIOpsState) -> dict[str, Any]:
    knowledge_count = len(state.get("knowledge_evidence", []))
    metric_count = len(state.get("metrics_evidence", []))
    log_count = len(state.get("log_evidence", []))
    evidence_count = knowledge_count + metric_count + log_count

    if evidence_count:
        summary = "已收集到运维证据，建议结合指标趋势、错误日志和知识库经验继续确认根因。"
        confidence = "low"
    else:
        summary = "当前未收集到有效运维证据，需要补充监控指标、日志或知识库信息后再判断。"
        confidence = "low"

    return {
        "summary": summary,
        "confidence": confidence,
        "source": "fallback",
        "evidence_count": evidence_count,
        "knowledge_evidence_count": knowledge_count,
        "metrics_evidence_count": metric_count,
        "log_evidence_count": log_count,
        "recommendations": [
            "确认告警时间窗口和受影响服务范围",
            "补充关键指标和错误日志证据",
            "对照知识库历史案例验证根因假设",
        ],
    }


def _diagnosis_context(state: MultiAgentAIOpsState) -> str:
    payload = {
        "input": state.get("input", ""),
        "knowledge_evidence": state.get("knowledge_evidence", []),
        "metrics_evidence": state.get("metrics_evidence", []),
        "log_evidence": state.get("log_evidence", []),
        "ops_evidence": state.get("ops_evidence", []),
        "previous_errors": state.get("errors", []),
    }
    return json.dumps(payload, ensure_ascii=False, default=str)


__all__ = ["diagnosis_agent"]
