from importlib import import_module


def _service_without_init():
    module = import_module("app.services.multi_agent_aiops_service")
    return module.MultiAgentAIOpsService.__new__(module.MultiAgentAIOpsService)


def test_format_plan_event_preserves_sse_contract():
    service = _service_without_init()

    event = service._format_plan_event({"plan": ["收集知识库经验", "查询监控证据"]})

    assert event["type"] == "plan"
    assert event["stage"] == "plan_created"
    assert event["plan"] == ["收集知识库经验", "查询监控证据"]
    assert "message" in event


def test_format_step_event_preserves_frontend_fields():
    service = _service_without_init()

    event = service._format_step_event(
        "knowledge_agent",
        {
            "current_stage": "knowledge_collection",
            "knowledge_evidence": [{"content": "CPU 告警处理经验"}],
        },
    )

    assert event["type"] == "step_complete"
    assert event["stage"] == "step_executed"
    assert event["current_step"] == "knowledge_agent"
    assert event["remaining_steps"] >= 0
    assert "message" in event


def test_format_report_event_keeps_markdown_report_field():
    service = _service_without_init()

    event = service._format_report_event({"report": "# 告警分析报告\n\n诊断完成"})

    assert event["type"] == "report"
    assert event["stage"] == "final_report"
    assert event["report"].startswith("# 告警分析报告")
    assert "message" in event


def test_format_complete_event_includes_diagnosis_report():
    service = _service_without_init()

    event = service._format_complete_event({"report": "# 告警分析报告\n\n诊断完成"})

    assert event["type"] == "complete"
    assert event["stage"] == "diagnosis_complete"
    assert event["diagnosis"]["status"] == "completed"
    assert event["diagnosis"]["report"].startswith("# 告警分析报告")
    assert "response" not in event


def test_format_error_event_terminates_stream_with_message():
    service = _service_without_init()

    event = service._format_error_event("exception", RuntimeError("boom"))

    assert event["type"] == "error"
    assert event["stage"] == "exception"
    assert "boom" in event["message"]
