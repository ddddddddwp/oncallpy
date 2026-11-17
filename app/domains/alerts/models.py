"""Alert domain data models and constants."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

StateType = Literal["new", "investigating", "mitigated", "closed"]

# 告警状态常量
STATE_NEW: StateType = "new"
STATE_INVESTIGATING: StateType = "investigating"
STATE_MITIGATED: StateType = "mitigated"
STATE_CLOSED: StateType = "closed"
ALERT_STATES: tuple[StateType, ...] = (
    STATE_NEW,
    STATE_INVESTIGATING,
    STATE_MITIGATED,
    STATE_CLOSED,
)

# 告警事件类型
EVENT_SOURCE_FIRED = "source_fired"
EVENT_SOURCE_RESOLVED = "source_resolved"
EVENT_STATE_CHANGE = "state_change"
EVENT_NOTE = "note"
EVENT_CLAIM = "claim"
EVENT_REASSIGN = "reassign"
EVENT_RELEASE = "release"
EVENT_ESCALATE = "escalate"
EVENT_SILENCE = "silence"

# 告警接入事件
INGEST_EVENT_FIRING = "firing"
INGEST_EVENT_UPDATE = "update"
INGEST_EVENT_RESOLVED = "resolved"

INGEST_ACTION_CREATED = "created"
INGEST_ACTION_UPDATED = "updated"
INGEST_ACTION_RESOLVED = "resolved"
INGEST_ACTION_DUPLICATE = "duplicate"


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Note(BaseModel):
    """告警备注."""

    author: str
    content: str
    created_at: datetime = Field(default_factory=_utcnow, alias="createdAt")

    model_config = ConfigDict(populate_by_name=True)


class TimelineEvent(BaseModel):
    """告警时间线事件."""

    type: str
    actor: str
    summary: str
    created_at: datetime = Field(default_factory=_utcnow, alias="createdAt")

    model_config = ConfigDict(populate_by_name=True)


class Alert(BaseModel):
    """告警实体."""

    id: str
    fingerprint: str | None = None
    source: str | None = None
    source_event_id: str | None = Field(default=None, alias="sourceEventId")
    source_alert_id: str | None = Field(default=None, alias="sourceAlertId")
    name: str = ""
    fault_code: str = Field(default="", alias="faultCode")
    fault_name: str = Field(default="", alias="faultName")
    fault_category: str = Field(default="", alias="faultCategory")
    service: str = ""
    severity: str = ""
    team: str = ""
    region: str = ""
    environment: str = ""
    device_id: str = Field(default="", alias="deviceId")
    product_key: str = Field(default="", alias="productKey")
    gateway_id: str = Field(default="", alias="gatewayId")
    metric_name: str = Field(default="", alias="metricName")
    metric_value: str = Field(default="", alias="metricValue")
    threshold: str = ""
    unit: str = ""
    source_state: str = Field(default="", alias="sourceState")
    state: str = ""
    status: str = ""
    active_at: datetime = Field(default_factory=_utcnow, alias="activeAt")
    ends_at: datetime | None = Field(default=None, alias="endsAt")
    last_seen_at: datetime = Field(default_factory=_utcnow, alias="lastSeenAt")
    resolved_at: datetime | None = Field(default=None, alias="resolvedAt")
    duration: str = ""
    description: str = ""
    labels: dict[str, str] = Field(default_factory=dict)
    annotations: dict[str, str] = Field(default_factory=dict)
    attributes: dict[str, Any] = Field(default_factory=dict)
    value: str = ""
    owner: str = ""
    owner_id: str = Field(default="", alias="ownerId")
    claimed_at: datetime | None = Field(default=None, alias="claimedAt")
    silenced: bool = False
    silence_id: str = Field(default="", alias="silenceId")
    silence_end: datetime | None = Field(default=None, alias="silenceEnd")
    notes: list[Note] = Field(default_factory=list)
    events: list[TimelineEvent] = Field(default_factory=list)
    event_count: int = Field(default=0, alias="eventCount")
    first_ingest_id: str = Field(default="", alias="firstIngestId")
    last_ingest_id: str = Field(default="", alias="lastIngestId")

    model_config = ConfigDict(populate_by_name=True)


class ExternalAlertPayload(BaseModel):
    """外部告警内容."""

    name: str = ""
    fault_code: str = Field(default="", alias="faultCode")
    fault_name: str = Field(default="", alias="faultName")
    fault_category: str = Field(default="", alias="faultCategory")
    severity: str = ""
    service: str = ""
    team: str = ""
    region: str = ""
    environment: str = ""
    device_id: str = Field(default="", alias="deviceId")
    product_key: str = Field(default="", alias="productKey")
    gateway_id: str = Field(default="", alias="gatewayId")
    metric_name: str = Field(default="", alias="metricName")
    metric_value: str = Field(default="", alias="metricValue")
    threshold: str = ""
    unit: str = ""
    description: str = ""
    starts_at: datetime = Field(default_factory=_utcnow, alias="startsAt")
    ends_at: datetime | None = Field(default=None, alias="endsAt")
    labels: dict[str, str] | None = None
    annotations: dict[str, str] | None = None
    attributes: dict[str, Any] | None = None

    model_config = ConfigDict(populate_by_name=True)


class AlertIngestInput(BaseModel):
    """告警接入输入."""

    source: str
    source_event_id: str = Field(default="", alias="sourceEventId")
    source_alert_id: str = Field(default="", alias="sourceAlertId")
    event_type: str = Field(default=INGEST_EVENT_FIRING, alias="eventType")
    actor: str = ""
    occurred_at: datetime | None = Field(default=None, alias="occurredAt")
    alert: ExternalAlertPayload
    raw: dict[str, Any] | None = None
    remote_addr: str = ""
    user_agent: str = ""

    model_config = ConfigDict(populate_by_name=True)


class AlertIngestResult(BaseModel):
    """告警接入返回结果."""

    ingest_id: str = Field(alias="ingestId")
    event_id: str = Field(alias="eventId")
    alert_id: str = Field(alias="alertId")
    action: str
    message: str

    model_config = ConfigDict(populate_by_name=True)


class AlertEvent(BaseModel):
    """告警事件流水."""

    event_id: str = Field(alias="eventId")
    ingest_id: str = Field(alias="ingestId")
    alert_id: str = Field(alias="alertId")
    fingerprint: str
    source: str
    source_event_id: str | None = Field(default=None, alias="sourceEventId")
    source_alert_id: str | None = Field(default=None, alias="sourceAlertId")
    event_type: str = Field(alias="eventType")
    fault_code: str = Field(default="", alias="faultCode")
    fault_name: str = Field(default="", alias="faultName")
    fault_category: str = Field(default="", alias="faultCategory")
    actor: str = ""
    summary: str = ""
    remote_addr: str = Field(default="", alias="remoteAddr")
    user_agent: str = Field(default="", alias="userAgent")
    occurred_at: datetime = Field(default_factory=_utcnow, alias="occurredAt")
    received_at: datetime = Field(default_factory=_utcnow, alias="receivedAt")
    before_state: dict[str, Any] | None = Field(default=None, alias="beforeState")
    after_state: dict[str, Any] | None = Field(default=None, alias="afterState")
    normalized_payload: dict[str, Any] | None = Field(default=None, alias="normalizedPayload")
    raw_payload: dict[str, Any] | None = Field(default=None, alias="rawPayload")
    created_at: datetime = Field(default_factory=_utcnow, alias="createdAt")

    model_config = ConfigDict(populate_by_name=True)


class SilenceMatcher(BaseModel):
    """静默规则匹配器."""

    field: str = ""
    label: str = ""
    operator: str
    value: str

    model_config = ConfigDict(populate_by_name=True)


class SilenceRule(BaseModel):
    """静默规则."""

    id: str
    name: str = ""
    matchers: list[SilenceMatcher] = Field(default_factory=list)
    starts_at: datetime = Field(alias="startsAt")
    ends_at: datetime = Field(alias="endsAt")
    created_by: str = Field(default="", alias="createdBy")
    created_at: datetime = Field(default_factory=_utcnow, alias="createdAt")
    updated_by: str = Field(default="", alias="updatedBy")
    updated_at: datetime = Field(default_factory=_utcnow, alias="updatedAt")
    comment: str = ""
    enabled: bool = True

    model_config = ConfigDict(populate_by_name=True)


class PrometheusAlert(BaseModel):
    """Prometheus 告警模型."""

    labels: dict[str, str] = Field(default_factory=dict)
    annotations: dict[str, str] = Field(default_factory=dict)
    state: str = ""
    active_at: str = Field(alias="activeAt")
    value: str = ""

    model_config = ConfigDict(populate_by_name=True)


class PrometheusAlertsResult(BaseModel):
    """Prometheus /api/v1/alerts 响应."""

    status: str = ""

    class Data(BaseModel):
        alerts: list[PrometheusAlert] = Field(default_factory=list)

    data: Data = Field(default_factory=Data)
    error: str = ""
    error_type: str = Field(default="", alias="errorType")

    model_config = ConfigDict(populate_by_name=True)
