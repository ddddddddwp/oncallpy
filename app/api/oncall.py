"""On-call 管理接口（原 oncall.go 迁移）。"""

from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from loguru import logger

from app.domains.oncall import (
    AlertEscalation,
    AlertGroup,
    AlertSignal,
    AggregationRule,
    AggregationResult,
    EscalationPolicy,
    EscalationRule,
    OnCallNotFound,
    OnCallResult,
    OnCallShift,
    OnCallStore,
    OnCallValidationError,
    RoutingDecision,
    RoutingRule,
    RouteMatcher,
    Schedule,
    TimeRestriction,
)

router = APIRouter()
_store = OnCallStore()


def _parse_optional_time(value: str | None) -> datetime:
    if not value:
        return datetime.min
    normalized = value.strip()
    if normalized == "":
        return datetime.min
    normalized = normalized.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail="time must be RFC3339",
        ) from exc


def _handle_error(exc: Exception) -> None:
    if isinstance(exc, OnCallNotFound):
        raise HTTPException(status_code=404, detail="值班资源不存在")
    if isinstance(exc, OnCallValidationError):
        raise HTTPException(status_code=400, detail=str(exc))
    logger.exception("oncall api error: %s", exc)
    raise HTTPException(status_code=500, detail=str(exc))


class EscalationPolicyCreateRequest(BaseModel):
    id: str = ""
    name: str
    description: str = ""
    team: str = ""
    rules: list[EscalationRule] = Field(default_factory=list)


class AlertEscalateRequest(BaseModel):
    policy_id: str = Field(default="", alias="policyId")
    level: int = 0
    actor: str = ""
    reason: str = ""
    targets: list[str] = Field(default_factory=list)


class AlertGroupCreateRequest(BaseModel):
    alert: AlertSignal | None = None
    id: str = ""
    name: str = ""
    service: str = ""
    severity: str = ""
    team: str = ""
    source: str = ""
    description: str = ""
    labels: dict[str, str] | None = None
    starts_at: str = Field(default="", alias="startsAt")


class AggregationRuleCreateRequest(BaseModel):
    id: str = ""
    name: str
    enabled: bool | None = None
    priority: int = 0
    group_by: list[str] = Field(default_factory=list, alias="groupBy")
    window_minutes: int = Field(default=0, alias="windowMinutes")
    matchers: list[RouteMatcher] = Field(default_factory=list)


class ScheduleCreateRequest(BaseModel):
    id: str = ""
    name: str
    team: str
    timezone: str = ""
    layers: list
    restrictions: list[TimeRestriction] = Field(default_factory=list)


class RoutingRuleCreateRequest(BaseModel):
    id: str = ""
    name: str
    enabled: bool | None = None
    priority: int = 0
    team: str = ""
    schedule_id: str = Field(default="", alias="scheduleId")
    escalation_policy_id: str = Field(default="", alias="escalationPolicyId")
    auto_assign: bool = Field(default=False, alias="autoAssign")
    matchers: list[RouteMatcher] = Field(default_factory=list)


class RoutingRuleUpdateRequest(RoutingRuleCreateRequest):
    id: str


class RoutingRuleMatchRequest(BaseModel):
    alert: AlertSignal


@router.get("/escalation-policies")
def escalation_policies_list() -> dict[str, list[EscalationPolicy]]:
    try:
        return {"policies": _store.list_escalation_policies()}
    except Exception as exc:  # pragma: no cover - defensive
        _handle_error(exc)
        raise


@router.post("/escalation-policies")
def escalation_policies_create(payload: EscalationPolicyCreateRequest) -> dict[str, EscalationPolicy]:
    try:
        policy = _store.upsert_escalation_policy(
            EscalationPolicy(
                id=payload.id,
                name=payload.name,
                description=payload.description,
                team=payload.team,
                rules=payload.rules,
            )
        )
        return {"policy": policy}
    except Exception as exc:
        _handle_error(exc)
        raise


@router.delete("/escalation-policies/{policy_id}")
def escalation_policies_delete(policy_id: str) -> dict[str, bool]:
    try:
        _store.delete_escalation_policy(policy_id)
        return {"deleted": True}
    except Exception as exc:
        _handle_error(exc)
        raise


@router.post("/alerts/{alert_id}/escalate")
def alerts_escalate(
    alert_id: str,
    payload: AlertEscalateRequest,
) -> dict[str, AlertEscalation]:
    try:
        escalation = _store.escalate_alert(
            alert_id=alert_id,
            manual={
                "policy_id": payload.policy_id,
                "level": payload.level,
                "actor": payload.actor,
                "reason": payload.reason,
                "targets": payload.targets,
            },
        )
        return {"escalation": escalation}
    except Exception as exc:
        _handle_error(exc)
        raise


@router.get("/alert-groups")
def alert_groups_list() -> dict[str, list[AlertGroup]]:
    try:
        return {"groups": _store.list_alert_groups()}
    except Exception as exc:
        _handle_error(exc)
        raise


@router.post("/alert-groups")
def alert_groups_create(
    payload: AlertGroupCreateRequest,
) -> dict[str, AggregationResult]:
    try:
        alert = payload.alert or AlertSignal()
        if payload.id:
            alert.id = payload.id
        if payload.name:
            alert.name = payload.name
        if payload.service:
            alert.service = payload.service
        if payload.severity:
            alert.severity = payload.severity
        if payload.team:
            alert.team = payload.team
        if payload.source:
            alert.source = payload.source
        if payload.description:
            alert.description = payload.description
        if payload.labels is not None:
            alert.labels = payload.labels
        if payload.starts_at:
            alert.starts_at = _parse_optional_time(payload.starts_at)
            if alert.starts_at == datetime.min:
                alert.starts_at = None
        result = _store.aggregate_alert(alert)
        return {"result": result}
    except Exception as exc:
        _handle_error(exc)
        raise


@router.get("/aggregation-rules")
def aggregation_rules_list() -> dict[str, list[AggregationRule]]:
    try:
        return {"rules": _store.list_aggregation_rules()}
    except Exception as exc:
        _handle_error(exc)
        raise


@router.post("/aggregation-rules")
def aggregation_rules_create(
    payload: AggregationRuleCreateRequest,
) -> dict[str, AggregationRule]:
    try:
        rule = _store.upsert_aggregation_rule(
            AggregationRule(
                id=payload.id,
                name=payload.name,
                enabled=payload.enabled if payload.enabled is not None else True,
                priority=payload.priority,
                group_by=payload.group_by,
                window_minutes=payload.window_minutes,
                matchers=payload.matchers,
            )
        )
        return {"rule": rule}
    except Exception as exc:
        _handle_error(exc)
        raise


@router.delete("/aggregation-rules/{rule_id}")
def aggregation_rules_delete(rule_id: str) -> dict[str, bool]:
    try:
        _store.delete_aggregation_rule(rule_id)
        return {"deleted": True}
    except Exception as exc:
        _handle_error(exc)
        raise


@router.get("/schedules")
def schedules_list(team: str = "") -> dict[str, list[Schedule]]:
    try:
        return {"schedules": _store.list_schedules(team=team)}
    except Exception as exc:
        _handle_error(exc)
        raise


@router.post("/schedules")
def schedules_create(payload: ScheduleCreateRequest) -> dict[str, Schedule]:
    try:
        schedule = _store.upsert_schedule(
            Schedule(
                id=payload.id,
                name=payload.name,
                team=payload.team,
                timezone=payload.timezone,
                layers=payload.layers,
                restrictions=payload.restrictions,
            )
        )
        return {"schedule": schedule}
    except Exception as exc:
        _handle_error(exc)
        raise


@router.delete("/schedules/{schedule_id}")
def schedules_delete(schedule_id: str) -> dict[str, bool]:
    try:
        _store.delete_schedule(schedule_id)
        return {"deleted": True}
    except Exception as exc:
        _handle_error(exc)
        raise


@router.get("/schedules/{schedule_id}/shifts")
def schedule_shifts(
    schedule_id: str,
    start: str = "",
    end: str = "",
    limit: int = 100,
) -> dict[str, list[OnCallShift]]:
    try:
        start_at = _parse_optional_time(start)
        if start_at == datetime.min:
            start_at = datetime.min
        end_at = _parse_optional_time(end)
        if end_at == datetime.min:
            end_at = datetime.min
        shifts = _store.schedule_shifts(
            schedule_id,
            start=start_at,
            end=end_at,
            limit=limit,
        )
        return {"shifts": shifts}
    except Exception as exc:
        _handle_error(exc)
        raise


@router.get("/schedules/{schedule_id}/current")
def schedule_current(
    schedule_id: str,
    at: str = "",
) -> dict[str, OnCallResult]:
    try:
        at_dt = _parse_optional_time(at)
        if at_dt == datetime.min:
            at_dt = datetime.min
        result = _store.current_on_call(schedule_id, at_dt)
        return {"result": result}
    except Exception as exc:
        _handle_error(exc)
        raise


@router.get("/oncall/{team}")
def oncall_by_team(team: str, at: str = "") -> dict[str, list[OnCallResult]]:
    try:
        at_dt = _parse_optional_time(at)
        if at_dt == datetime.min:
            at_dt = datetime.min
        results = _store.current_on_call_by_team(team, at_dt)
        return {"results": results}
    except Exception as exc:
        _handle_error(exc)
        raise


@router.get("/routing-rules")
def routing_rules_list() -> dict[str, list[RoutingRule]]:
    try:
        return {"rules": _store.list_routing_rules()}
    except Exception as exc:
        _handle_error(exc)
        raise


@router.post("/routing-rules")
def routing_rules_create(payload: RoutingRuleCreateRequest) -> dict[str, RoutingRule]:
    try:
        rule = _store.upsert_routing_rule(
            RoutingRule(
                id=payload.id,
                name=payload.name,
                enabled=payload.enabled if payload.enabled is not None else True,
                priority=payload.priority,
                team=payload.team,
                schedule_id=payload.schedule_id,
                escalation_policy_id=payload.escalation_policy_id,
                auto_assign=payload.auto_assign,
                matchers=payload.matchers,
            )
        )
        return {"rule": rule}
    except Exception as exc:
        _handle_error(exc)
        raise


@router.put("/routing-rules/{rule_id}")
def routing_rules_update(
    rule_id: str,
    payload: RoutingRuleUpdateRequest,
) -> dict[str, RoutingRule]:
    try:
        payload.id = rule_id
        rule = _store.update_routing_rule(
            RoutingRule(
                id=payload.id,
                name=payload.name,
                enabled=payload.enabled if payload.enabled is not None else True,
                priority=payload.priority,
                team=payload.team,
                schedule_id=payload.schedule_id,
                escalation_policy_id=payload.escalation_policy_id,
                auto_assign=payload.auto_assign,
                matchers=payload.matchers,
            )
        )
        return {"rule": rule}
    except Exception as exc:
        _handle_error(exc)
        raise


@router.delete("/routing-rules/{rule_id}")
def routing_rules_delete(rule_id: str) -> dict[str, bool]:
    try:
        _store.delete_routing_rule(rule_id)
        return {"deleted": True}
    except Exception as exc:
        _handle_error(exc)
        raise


@router.post("/routing-rules/match")
def routing_rules_match(payload: RoutingRuleMatchRequest) -> dict[str, RoutingDecision]:
    try:
        decision = _store.route_alert(alert=payload.alert, at=payload.alert.starts_at)
        return {"decision": decision}
    except Exception as exc:
        _handle_error(exc)
        raise
