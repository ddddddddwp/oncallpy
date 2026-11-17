"""Utility helpers for on-call models, matching, and schedule math."""

from __future__ import annotations

import hashlib
import random
import re
from datetime import datetime, timezone

from .errors import OnCallValidationError
from .models import AlertSignal, RouteMatcher


_SAFE_FLOAT_TZ_RE = re.compile(r"^[+-]?([0-9]|1[0-1]):[0-5][0-9]$")


def new_id(prefix: str) -> str:
    """Generate short random identifiers with a domain prefix."""

    random_bits = random.getrandbits(64)
    return f"{prefix}_{random_bits:016x}"


def stable_id(prefix: str, value: str) -> str:
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{digest}"


def compact_strings(values: list[str] | tuple[str, ...] | None) -> list[str]:
    if not values:
        return []
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        trimmed = value.strip()
        if trimmed == "":
            continue
        if trimmed in seen:
            continue
        seen.add(trimmed)
        out.append(trimmed)
    return out


def compact_first_non_empty(*values: str | None) -> str:
    for value in values:
        if isinstance(value, str):
            normalized = value.strip()
            if normalized != "":
                return normalized
    return ""


first_non_blank = compact_first_non_empty


def normalize_weekdays(values: list[int] | None) -> list[int]:
    if not values:
        return []
    normalized: list[int] = []
    seen: set[int] = set()
    for value in values:
        if not isinstance(value, int):
            continue
        weekday = value % 7
        if weekday < 0:
            weekday += 7
        if weekday in seen:
            continue
        seen.add(weekday)
        normalized.append(weekday)
    return normalized


def normalize_matchers(matchers: list[RouteMatcher] | None) -> list[RouteMatcher]:
    if not matchers:
        return []
    return [
        RouteMatcher(
            field=matcher.field.strip(),
            operator=matcher.operator.strip(),
            value=matcher.value.strip(),
            values=compact_strings(matcher.values),
        )
        for matcher in matchers
    ]


def normalize_alert_signal(alert: AlertSignal) -> AlertSignal:
    labels = dict(alert.labels or {})

    name = compact_first_non_empty(
        alert.name,
        labels.get("alertname"),
        labels.get("name"),
    )
    service = compact_first_non_empty(
        alert.service,
        labels.get("service"),
        labels.get("job"),
        labels.get("app"),
    )
    severity = compact_first_non_empty(
        alert.severity,
        labels.get("severity"),
        labels.get("priority"),
    )
    team = compact_first_non_empty(
        alert.team,
        labels.get("team"),
        labels.get("owner"),
    )
    source = compact_first_non_empty(
        alert.source,
        labels.get("source"),
    )

    if name != "":
        labels["alertname"] = name
    if service != "":
        labels["service"] = service
    if severity != "":
        labels["severity"] = severity
    if team != "":
        labels["team"] = team

    return AlertSignal(
        id=alert.id.strip(),
        name=name,
        service=service,
        severity=severity,
        team=team,
        source=source,
        description=(alert.description or "").strip(),
        labels=labels,
        starts_at=ensure_utc(alert.starts_at),
    )


def alert_field_value(alert: AlertSignal, field: str) -> tuple[str, bool]:
    alert = normalize_alert_signal(alert)
    normalized_field = field.strip().lower()
    if normalized_field in {"id", "alert_id", "alertid"}:
        return value_exists(alert.id)
    if normalized_field in {"name", "alertname", "alert_name"}:
        return value_exists(alert.name)
    if normalized_field == "service":
        return value_exists(alert.service)
    if normalized_field == "severity":
        return value_exists(alert.severity)
    if normalized_field == "team":
        return value_exists(alert.team)
    if normalized_field == "source":
        return value_exists(alert.source)
    if normalized_field == "description":
        return value_exists(alert.description)

    label_key = normalized_field
    if normalized_field.startswith("labels."):
        label_key = normalized_field[len("labels."):]
    elif normalized_field.startswith("label:"):
        label_key = normalized_field[len("label:"):]
    return value_exists(alert.labels.get(label_key, ""))


def value_exists(value: str) -> tuple[str, bool]:
    normalized = value.strip()
    return normalized, normalized != ""


def is_false_matcher_value(value: str) -> bool:
    return value.strip().lower() in {"false", "0", "no"}


def match_one(matcher: RouteMatcher, alert: AlertSignal) -> bool:
    field_value, exists = alert_field_value(alert, matcher.field)
    operator = matcher.operator.strip().lower()
    if operator == "":
        operator = "equals"
    value = matcher.value.strip()

    if operator in {"equals", "eq", "="}:
        return exists and field_value == value
    if operator in {"not_equals", "ne", "!="}:
        return (not exists) or field_value != value
    if operator == "contains":
        return exists and value in field_value
    if operator == "regex":
        if not exists:
            return False
        try:
            return re.search(value, field_value) is not None
        except re.error:
            return False
    if operator == "in":
        if not exists:
            return False
        for candidate in compact_strings(matcher.values):
            if field_value == candidate:
                return True
        if value:
            for candidate in compact_strings(part for part in value.split(",")):
                if field_value == candidate:
                    return True
        return False
    if operator == "exists":
        if is_false_matcher_value(value):
            return not exists
        return exists
    return False


def match_all(matchers: list[RouteMatcher], alert: AlertSignal) -> bool:
    for matcher in matchers:
        if not match_one(matcher, alert):
            return False
    return True


def normalize_start_minute(minute: int) -> int:
    if minute < 0:
        return 0
    if minute >= 24 * 60:
        return 24 * 60 - 1
    return minute


def normalize_end_minute(minute: int) -> int:
    if minute < 0:
        return 0
    if minute > 24 * 60:
        return 24 * 60
    return minute


def weekday_allowed(weekdays: list[int] | None, weekday: int) -> bool:
    if not weekdays:
        return True
    return weekday in normalize_weekdays(weekdays)


def ensure_utc(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def parse_rfc3339(value: str | None) -> datetime | None:
    if value in (None, ""):
        return None
    normalized = value.strip()
    if normalized.endswith("Z"):
        normalized = f"{normalized[:-1]}+00:00"
    try:
        return datetime.fromisoformat(normalized)
    except ValueError as error:
        raise OnCallValidationError("time must be RFC3339") from error


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def min_time(a: datetime, b: datetime) -> datetime:
    return a if a <= b else b


def max_time(a: datetime, b: datetime) -> datetime:
    return a if a >= b else b
