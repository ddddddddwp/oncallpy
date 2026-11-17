"""Domain models for on-call scheduling and routing domain."""

from __future__ import annotations

from datetime import datetime, timezone
from pydantic import BaseModel, ConfigDict, Field

ESCALATION_STATUS_TRIGGERED = "triggered"
GROUP_STATUS_OPEN = "open"

MATCHER_EQUALS = "equals"
MATCHER_NOT_EQUALS = "not_equals"
MATCHER_CONTAINS = "contains"
MATCHER_REGEX = "regex"
MATCHER_EXISTS = "exists"
MATCHER_IN = "in"


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class ManualEscalation(BaseModel):
    policy_id: str = Field(default="", alias="policyId")
    level: int = 0
    actor: str = ""
    reason: str = ""
    targets: list[str] = Field(default_factory=list)

    model_config = ConfigDict(populate_by_name=True, extra="ignore")


class EscalationRule(BaseModel):
    id: str = ""
    level: int = 0
    delay_minutes: int = Field(default=0, alias="delayMinutes")
    notify_targets: list[str] = Field(default_factory=list, alias="notifyTargets")
    repeat_minutes: int = Field(default=0, alias="repeatMinutes")
    stop_on_acknowledged: bool = Field(default=False, alias="stopOnAcknowledged")

    model_config = ConfigDict(populate_by_name=True, extra="ignore")


class EscalationPolicy(BaseModel):
    id: str = ""
    name: str = ""
    description: str = ""
    team: str = ""
    rules: list[EscalationRule] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=_utcnow, alias="createdAt")
    updated_at: datetime = Field(default_factory=_utcnow, alias="updatedAt")

    model_config = ConfigDict(populate_by_name=True, extra="ignore")


class AlertEscalation(BaseModel):
    id: str = ""
    alert_id: str = Field(alias="alertId")
    policy_id: str = Field(default="", alias="policyId")
    level: int = 0
    actor: str = ""
    reason: str = ""
    targets: list[str] = Field(default_factory=list)
    status: str = ""
    created_at: datetime = Field(default_factory=_utcnow, alias="createdAt")
    updated_at: datetime = Field(default_factory=_utcnow, alias="updatedAt")

    model_config = ConfigDict(populate_by_name=True, extra="ignore")


class AlertSignal(BaseModel):
    id: str = ""
    name: str = ""
    service: str = ""
    severity: str = ""
    team: str = ""
    source: str = ""
    description: str = ""
    labels: dict[str, str] = Field(default_factory=dict)
    starts_at: datetime | None = Field(default=None, alias="startsAt")

    model_config = ConfigDict(populate_by_name=True, extra="ignore")


class AlertGroup(BaseModel):
    id: str = ""
    rule_id: str = Field(alias="ruleId")
    dedupe_key: str = Field(alias="dedupeKey")
    name: str = ""
    alert_ids: list[str] = Field(default_factory=list, alias="alertIds")
    fingerprints: list[str] = Field(default_factory=list)
    labels: dict[str, str] | None = None
    service: str = ""
    severity: str = ""
    team: str = ""
    owner: str = ""
    status: str = GROUP_STATUS_OPEN
    count: int = 0
    first_seen: datetime = Field(default_factory=_utcnow, alias="firstSeen")
    last_seen: datetime = Field(default_factory=_utcnow, alias="lastSeen")
    created_at: datetime = Field(default_factory=_utcnow, alias="createdAt")
    updated_at: datetime = Field(default_factory=_utcnow, alias="updatedAt")

    model_config = ConfigDict(populate_by_name=True, extra="ignore")


class AggregationRule(BaseModel):
    id: str = ""
    name: str = ""
    enabled: bool = True
    priority: int = 0
    group_by: list[str] = Field(default_factory=list, alias="groupBy")
    window_minutes: int = Field(default=0, alias="windowMinutes")
    matchers: list["RouteMatcher"] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=_utcnow, alias="createdAt")
    updated_at: datetime = Field(default_factory=_utcnow, alias="updatedAt")

    model_config = ConfigDict(populate_by_name=True, extra="ignore")


class AggregationResult(BaseModel):
    group: AlertGroup
    new_group: bool = Field(alias="newGroup")
    duplicate: bool = False
    routing: "RoutingDecision | None" = None

    model_config = ConfigDict(populate_by_name=True, extra="ignore")


class TimeRestriction(BaseModel):
    weekdays: list[int] = Field(default_factory=list)
    start_minute: int = Field(default=0, alias="startMinute")
    end_minute: int = Field(default=0, alias="endMinute")

    model_config = ConfigDict(populate_by_name=True, extra="ignore")


class ScheduleLayer(BaseModel):
    id: str = ""
    name: str = ""
    users: list[str] = Field(default_factory=list)
    start: datetime = Field(default_factory=_utcnow)
    end: datetime | None = None
    rotation_minutes: int = Field(default=0, alias="rotationMinutes")
    restrictions: list[TimeRestriction] = Field(default_factory=list)

    model_config = ConfigDict(populate_by_name=True, extra="ignore")


class Schedule(BaseModel):
    id: str = ""
    name: str = ""
    team: str = ""
    timezone: str = ""
    layers: list[ScheduleLayer] = Field(default_factory=list)
    restrictions: list[TimeRestriction] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=_utcnow, alias="createdAt")
    updated_at: datetime = Field(default_factory=_utcnow, alias="updatedAt")

    model_config = ConfigDict(populate_by_name=True, extra="ignore")


class OnCallShift(BaseModel):
    id: str = ""
    schedule_id: str = Field(alias="scheduleId")
    schedule_name: str = Field(default="", alias="scheduleName")
    team: str = ""
    layer_id: str = Field(alias="layerId")
    layer_name: str = Field(default="", alias="layerName")
    user: str = ""
    start: datetime
    end: datetime
    rotation_minutes: int = Field(alias="rotationMinutes")

    model_config = ConfigDict(populate_by_name=True, extra="ignore")


class OnCallResult(BaseModel):
    schedule_id: str = Field(alias="scheduleId")
    schedule_name: str = Field(alias="scheduleName")
    team: str
    timezone: str
    at: datetime
    users: list[str] = Field(default_factory=list)
    shifts: list[OnCallShift] = Field(default_factory=list)

    model_config = ConfigDict(populate_by_name=True, extra="ignore")


class RoutingRule(BaseModel):
    id: str = ""
    name: str = ""
    enabled: bool = True
    priority: int = 0
    team: str = ""
    schedule_id: str = Field(default="", alias="scheduleId")
    escalation_policy_id: str = Field(default="", alias="escalationPolicyId")
    auto_assign: bool = Field(default=False, alias="autoAssign")
    matchers: list["RouteMatcher"] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=_utcnow, alias="createdAt")
    updated_at: datetime = Field(default_factory=_utcnow, alias="updatedAt")

    model_config = ConfigDict(populate_by_name=True, extra="ignore")


class RouteMatcher(BaseModel):
    field: str = ""
    operator: str = ""
    value: str = ""
    values: list[str] = Field(default_factory=list)

    model_config = ConfigDict(populate_by_name=True, extra="ignore")


class RoutingDecision(BaseModel):
    matched: bool = False
    rule_id: str = Field(default="", alias="ruleId")
    rule_name: str = Field(default="", alias="ruleName")
    team: str = ""
    schedule_id: str = Field(default="", alias="scheduleId")
    escalation_policy_id: str = Field(default="", alias="escalationPolicyId")
    assignee: str = ""
    matched_at: datetime | None = Field(default=None, alias="matchedAt")

    model_config = ConfigDict(populate_by_name=True, extra="ignore")


class OnCallState(BaseModel):
    escalation_policies: dict[str, EscalationPolicy] = Field(
        default_factory=dict,
        alias="escalationPolicies",
    )
    alert_escalations: dict[str, AlertEscalation] = Field(
        default_factory=dict,
        alias="alertEscalations",
    )
    aggregation_rules: dict[str, AggregationRule] = Field(
        default_factory=dict,
        alias="aggregationRules",
    )
    alert_groups: dict[str, AlertGroup] = Field(
        default_factory=dict,
        alias="alertGroups",
    )
    schedules: dict[str, Schedule] = Field(default_factory=dict)
    routing_rules: dict[str, RoutingRule] = Field(
        default_factory=dict,
        alias="routingRules",
    )

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    def model_dump(self, **kwargs) -> dict[str, object]:  # type: ignore[override]
        kwargs.setdefault("by_alias", True)
        return super().model_dump(**kwargs)
