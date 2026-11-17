"""
Multi-agent AIOps service.
"""

from __future__ import annotations

from textwrap import dedent
from typing import Any, AsyncGenerator

from langgraph.graph import END, StateGraph
from loguru import logger

from app.core.checkpoint import close_checkpointer, create_redis_checkpointer


NODE_PLANNER = "planner"
NODE_KNOWLEDGE_AGENT = "knowledge_agent"
NODE_OPS_EVIDENCE_AGENT = "ops_evidence_agent"
NODE_DIAGNOSIS_AGENT = "diagnosis_agent"
NODE_REPORT_AGENT = "report_agent"


class MultiAgentAIOpsService:
    """Linear multi-agent AIOps diagnosis workflow service."""

    def __init__(self) -> None:
        self.checkpointer = None
        self.graph = None
        self._initialization_error: Exception | None = None
        logger.info("Multi-agent AIOps Service 初始化完成")

    async def initialize(self) -> None:
        """启动时初始化 Redis checkpointer 和工作流图。"""
        await self._ensure_graph()

    async def cleanup(self) -> None:
        """关闭 Redis checkpointer 资源。"""
        if self.checkpointer is not None:
            await close_checkpointer(self.checkpointer)
            self.checkpointer = None
        self.graph = None

    def _build_graph(self):
        """Build planner -> knowledge -> ops evidence -> diagnosis -> report -> END."""
        from app.agent.aiops_multi import (
            MultiAgentAIOpsState,
            diagnosis_agent,
            knowledge_agent,
            ops_evidence_agent,
            planner,
            report_agent,
        )

        logger.info("构建 Multi-agent AIOps 工作流图...")

        workflow = StateGraph(MultiAgentAIOpsState)
        workflow.add_node(NODE_PLANNER, planner)
        workflow.add_node(NODE_KNOWLEDGE_AGENT, knowledge_agent)
        workflow.add_node(NODE_OPS_EVIDENCE_AGENT, ops_evidence_agent)
        workflow.add_node(NODE_DIAGNOSIS_AGENT, diagnosis_agent)
        workflow.add_node(NODE_REPORT_AGENT, report_agent)

        workflow.set_entry_point(NODE_PLANNER)
        workflow.add_edge(NODE_PLANNER, NODE_KNOWLEDGE_AGENT)
        workflow.add_edge(NODE_KNOWLEDGE_AGENT, NODE_OPS_EVIDENCE_AGENT)
        workflow.add_edge(NODE_OPS_EVIDENCE_AGENT, NODE_DIAGNOSIS_AGENT)
        workflow.add_edge(NODE_DIAGNOSIS_AGENT, NODE_REPORT_AGENT)
        workflow.add_edge(NODE_REPORT_AGENT, END)

        compiled_graph = workflow.compile(checkpointer=self.checkpointer)
        logger.info("Multi-agent AIOps 工作流图构建完成")
        return compiled_graph

    async def diagnose(
        self,
        session_id: str = "default",
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Run the default AIOps diagnosis task."""
        aiops_task = dedent(
            """诊断当前系统是否存在告警，如果存在告警请详细分析告警原因并生成诊断报告，诊断报告输出格式要求：
                ```
                # 告警分析报告

                ---

                ## 📋 活跃告警清单

                | 告警名称 | 级别 | 目标服务 | 首次触发时间 | 最新触发时间 | 状态 |
                |---------|------|----------|-------------|-------------|------|
                | [告警1名称] | [级别] | [服务名] | [时间] | [时间] | 活跃 |
                | [告警2名称] | [级别] | [服务名] | [时间] | [时间] | 活跃 |

                ---

                ## 🔍 告警根因分析1 - [告警名称]

                ### 告警详情
                - **告警级别**: [级别]
                - **受影响服务**: [服务名]
                - **持续时间**: [X分钟]

                ### 症状描述
                [根据监控指标描述症状]

                ### 日志证据
                [引用查询到的关键日志]

                ### 根因结论
                [基于证据得出的根本原因]

                ---

                ## 🛠️ 处理方案执行1 - [告警名称]

                ### 已执行的排查步骤
                1. [步骤1]
                2. [步骤2]

                ### 处理建议
                [给出具体的处理建议]

                ### 预期效果
                [说明预期的效果]

                ---

                ## 🔍 告警根因分析2 - [告警名称]
                [如果有第2个告警，重复上述格式]

                ---

                ## 📊 结论

                ### 整体评估
                [总结所有告警的整体情况]

                ### 关键发现
                - [发现1]
                - [发现2]

                ### 后续建议
                1. [建议1]
                2. [建议2]

                ### 风险评估
                [评估当前风险等级和影响范围]
                ```

                **重要提醒**：
                - 最终输出必须是纯 Markdown 文本，不要包含 JSON 结构
                - 所有内容必须基于工具查询的真实数据，严禁编造
                - 如果某个步骤失败，在结论中如实说明，不要跳过"""
        )

        async for event in self.execute(aiops_task, session_id=session_id):
            yield event

    async def execute(
        self,
        user_input: str,
        session_id: str = "default",
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Execute the multi-agent workflow and stream contract-compatible events."""
        logger.info(f"[会话 {session_id}] 开始执行 Multi-agent AIOps 任务")

        try:
            graph = await self._ensure_graph()
            initial_state = self._build_initial_state(user_input, session_id)
            config_dict = {"configurable": {"thread_id": session_id}}
            state_snapshot: dict[str, Any] = dict(initial_state)

            async for event in graph.astream(
                input=initial_state,
                config=config_dict,
                stream_mode="updates",
            ):
                for node_name, node_output in event.items():
                    node_state = self._as_dict(node_output)
                    self._merge_state_update(state_snapshot, node_state)
                    logger.info(f"Multi-agent AIOps 节点 '{node_name}' 输出事件")

                    if node_name == NODE_PLANNER:
                        yield self._format_plan_event(node_state)
                    elif node_name == NODE_REPORT_AGENT:
                        yield self._format_report_event(state_snapshot)
                    else:
                        yield self._format_step_event(node_name, state_snapshot)

            final_state = await self._get_final_state(graph, config_dict, state_snapshot)
            yield self._format_complete_event(final_state)
            logger.info(f"[会话 {session_id}] Multi-agent AIOps 任务执行完成")

        except Exception as exc:
            logger.error(f"[会话 {session_id}] Multi-agent AIOps 任务执行失败: {exc}", exc_info=True)
            yield self._format_error_event("exception", exc)

    async def _ensure_graph(self):
        if getattr(self, "graph", None) is not None:
            return self.graph

        if getattr(self, "checkpointer", None) is None:
            self.checkpointer = await create_redis_checkpointer()

        self.graph = self._build_graph()
        self._initialization_error = None
        return self.graph

    def _build_initial_state(self, user_input: str, session_id: str) -> dict[str, Any]:
        return {
            "input": user_input,
            "user_input": user_input,
            "session_id": session_id,
            "plan": [],
            "past_steps": [],
            "knowledge_evidence": [],
            "ops_evidence": [],
            "metrics_evidence": [],
            "log_evidence": [],
            "tool_results": [],
            "diagnosis": {},
            "recommendations": [],
            "errors": [],
            "report": "",
            "response": "",
            "current_stage": "initialized",
        }

    async def _get_final_state(
        self,
        graph: Any,
        config_dict: dict[str, Any],
        fallback_state: dict[str, Any],
    ) -> dict[str, Any]:
        try:
            if hasattr(graph, "aget_state"):
                final_state = await graph.aget_state(config_dict)
            else:
                final_state = graph.get_state(config_dict)
        except Exception as exc:
            logger.warning(f"获取 Multi-agent AIOps 最终状态失败，使用流式快照: {exc}")
            return fallback_state

        if final_state and getattr(final_state, "values", None):
            return self._as_dict(final_state.values)

        return fallback_state

    def _format_plan_event(self, state: dict[str, Any] | None) -> dict[str, Any]:
        """Format a planner update while preserving the existing SSE contract."""
        if not state:
            return {
                "type": "status",
                "stage": "planner",
                "message": "规划节点执行中",
            }

        plan = self._as_list(state.get("plan"))
        return {
            "type": "plan",
            "stage": "plan_created",
            "message": f"诊断计划已制定，共 {len(plan)} 个步骤",
            "plan": plan,
        }

    def _format_step_event(
        self,
        node_name: str,
        state: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Format an intermediate agent update."""
        state = state or {}
        plan = self._as_list(state.get("plan"))
        past_steps = self._as_list(state.get("past_steps"))
        remaining_steps = state.get("remaining_steps")
        if not isinstance(remaining_steps, int):
            remaining_steps = max(len(plan) - len(past_steps), 0)

        event: dict[str, Any] = {
            "type": "step_complete",
            "stage": "step_executed",
            "message": f"{node_name} 执行完成",
            "current_step": node_name,
            "remaining_steps": remaining_steps,
        }

        for key in (
            "current_stage",
            "knowledge_evidence",
            "ops_evidence",
            "metrics_evidence",
            "log_evidence",
            "tool_results",
            "diagnosis",
            "recommendations",
            "errors",
            "result_preview",
        ):
            if key in state:
                event[key] = state[key]

        return event

    def _format_report_event(self, state: dict[str, Any] | None) -> dict[str, Any]:
        """Format the final report event. The report field is required by the frontend."""
        state = state or {}
        report = self._extract_report(state)
        event: dict[str, Any] = {
            "type": "report",
            "stage": "final_report",
            "message": "最终诊断报告已生成",
            "report": report,
        }

        evidence = {
            "knowledge": state.get("knowledge_evidence"),
            "ops": state.get("ops_evidence"),
            "diagnosis": state.get("diagnosis"),
        }
        event["evidence"] = {key: value for key, value in evidence.items() if value}
        return event

    def _format_complete_event(self, state: dict[str, Any] | None) -> dict[str, Any]:
        """Format terminal completion without a top-level response field."""
        state = state or {}
        report = self._extract_report(state)
        diagnosis = state.get("diagnosis")

        if isinstance(diagnosis, dict):
            diagnosis_payload = dict(diagnosis)
            diagnosis_payload.setdefault("status", "completed")
            diagnosis_payload["report"] = report
        else:
            diagnosis_payload = {
                "status": "completed",
                "report": report,
            }

        return {
            "type": "complete",
            "stage": "diagnosis_complete",
            "message": "诊断流程完成",
            "diagnosis": diagnosis_payload,
        }

    def _format_error_event(self, stage: str, error: Exception | str) -> dict[str, Any]:
        """Format a terminal error event."""
        message = str(error)
        return {
            "type": "error",
            "stage": stage,
            "message": f"诊断过程发生错误: {message}",
        }

    def _extract_report(self, state: dict[str, Any]) -> str:
        report = state.get("report")
        if isinstance(report, str) and report:
            return report

        diagnosis = state.get("diagnosis")
        if isinstance(diagnosis, dict):
            diagnosis_report = diagnosis.get("report")
            if isinstance(diagnosis_report, str) and diagnosis_report:
                return diagnosis_report

        response = state.get("response")
        if isinstance(response, str) and response:
            return response

        if isinstance(report, str):
            return report

        return ""

    def _as_dict(self, value: Any) -> dict[str, Any]:
        if isinstance(value, dict):
            return value
        if hasattr(value, "model_dump"):
            dumped = value.model_dump()
            return dumped if isinstance(dumped, dict) else {}
        return {}

    def _as_list(self, value: Any) -> list[Any]:
        return value if isinstance(value, list) else []

    def _merge_state_update(
        self,
        state_snapshot: dict[str, Any],
        node_state: dict[str, Any],
    ) -> None:
        append_fields = {
            "past_steps",
            "knowledge_evidence",
            "ops_evidence",
            "metrics_evidence",
            "log_evidence",
            "tool_results",
            "errors",
        }
        for key, value in node_state.items():
            if key in append_fields and isinstance(value, list):
                existing = state_snapshot.get(key)
                if isinstance(existing, list):
                    existing.extend(value)
                else:
                    state_snapshot[key] = list(value)
            else:
                state_snapshot[key] = value


multi_agent_aiops_service = MultiAgentAIOpsService()
