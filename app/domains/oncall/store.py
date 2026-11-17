"""Pure-Python implementation of on-call domain behavior."""

from __future__ import annotations

import hashlib
import os
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import RLock
from typing import Any
from zoneinfo import ZoneInfo

from app.domains.shared.atomic_json_repository import AtomicJsonRepository

from .errors import OnCallNotFound, OnCallValidationError
from .models import (
    AggregationResult,
    AggregationRule,
    AlertEscalation,
    AlertGroup,
    AlertSignal,
    EscalationPolicy,
    EscalationRule,
    GROUP_STATUS_OPEN,
    ManualEscalation,
    OnCallResult,
    OnCallShift,
    OnCallState,
    RoutingDecision,
    RoutingRule,
    Schedule,
    ScheduleLayer,
    TimeRestriction,
)
from .utils import (
    alert_field_value,
    compact_strings,
    ensure_utc,
    first_non_blank,
    match_all,
    max_time,
    min_time,
    new_id,
    normalize_alert_signal,
    normalize_end_minute,
    normalize_matchers,
    normalize_start_minute,
    stable_id,
    weekday_allowed,
)


# Re-export for compatibility with references that import it from store.
OnCallValidationError = OnCallValidationError


class OnCallStore:
    """Thread-safe stateful store for on-call configuration and data."""

    def __init__(
        self,
        state_file: str | Path | None = None,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        if state_file:
            path = Path(state_file)
        else:
            path = _default_state_file()

        self._state_file = path
        self._state = OnCallState()
        self._state_lock = RLock()
        self._now = now or self._utc_now
        self._repository = AtomicJsonRepository(
            self._state_file,
            lambda: self._state.model_dump(),
        )
        self._load_state()

    def list_escalation_policies(self) -> list[EscalationPolicy]:
        with self._state_lock:
            items = [policy.model_copy(deep=True) for policy in self._state.escalation_policies.values()]
        return sorted(items, key=lambda policy: (policy.name, policy.id))

    def upsert_escalation_policy(self, policy: EscalationPolicy) -> EscalationPolicy:
        policy = EscalationPolicy.model_validate(policy)
        policy.id = policy.id.strip()
        policy.name = policy.name.strip()
        policy.description = policy.description.strip()
        policy.team = policy.team.strip()
        if policy.name == "":
            raise OnCallValidationError("escalation policy name is required")

        seen_levels: set[int] = set()
        normalized_rules: list[EscalationRule] = []
        for index, rule in enumerate(policy.rules):
            level = int(rule.level)
            if rule.delay_minutes < 0:
                raise OnCallValidationError("escalation rule delay must be non-negative")
            if rule.repeat_minutes < 0:
                raise OnCallValidationError("escalation rule repeat must be non-negative")

            if level <= 0:
                level = index + 1
            if level in seen_levels:
                raise OnCallValidationError(f"duplicate escalation rule level {level}")
            seen_levels.add(level)

            normalized_rules.append(
                EscalationRule(
                    id=rule.id.strip() or new_id("er"),
                    level=level,
                    delay_minutes=rule.delay_minutes,
                    notify_targets=compact_strings(rule.notify_targets),
                    repeat_minutes=rule.repeat_minutes,
                    stop_on_acknowledged=rule.stop_on_acknowledged,
                )
            )

        normalized_rules.sort(key=lambda rule: rule.level)

        now = self._utc_now()
        with self._state_lock:
            existing = self._state.escalation_policies.get(policy.id) if policy.id else None
            policy_payload = EscalationPolicy(
                id=policy.id or new_id("ep"),
                name=policy.name,
                description=policy.description,
                team=policy.team,
                rules=normalized_rules,
                created_at=existing.created_at if existing else now,
                updated_at=now,
            )
            self._state.escalation_policies[policy_payload.id] = policy_payload
            self._save_state_locked()
            return policy_payload.model_copy(deep=True)

    def delete_escalation_policy(self, policy_id: str) -> None:
        policy_id = policy_id.strip()
        if policy_id == "":
            raise OnCallValidationError("escalation policy id is required")

        with self._state_lock:
            if policy_id not in self._state.escalation_policies:
                raise OnCallNotFound(f"escalation policy {policy_id}")
            ref = self._first_routing_rule_for_policy_locked(policy_id)
            if ref:
                raise OnCallValidationError(
                    f"escalation policy {policy_id} is used by routing rule {ref}"
                )
            del self._state.escalation_policies[policy_id]
            self._save_state_locked()

    def escalate_alert(self, alert_id: str, manual: ManualEscalation | dict[str, Any] | object) -> AlertEscalation:
        if isinstance(manual, ManualEscalation):
            manual = manual.model_dump(mode="python")
        manual = ManualEscalation.model_validate(manual)

        alert_id = alert_id.strip()
        if alert_id == "":
            raise OnCallValidationError("alert id is required")

        with self._state_lock:
            now = self._utc_now()
            level = int(manual.level)
            if level <= 0:
                level = self._next_escalation_level_locked(alert_id)

            policy_id = manual.policy_id.strip()
            targets = compact_strings(manual.targets)
            if policy_id:
                policy = self._state.escalation_policies.get(policy_id)
                if policy is None:
                    raise OnCallNotFound(f"escalation policy {policy_id}")
                if not targets:
                    targets = self._targets_for_escalation_level(policy, level)

            if not targets:
                raise OnCallValidationError("escalation targets are required")

            escalation = AlertEscalation(
                id=new_id("esc"),
                alert_id=alert_id,
                policy_id=policy_id,
                level=level,
                actor=(manual.actor or "system").strip() or "system",
                reason=manual.reason.strip(),
                targets=targets,
                status="triggered",
                created_at=now,
                updated_at=now,
            )
            self._state.alert_escalations[escalation.id] = escalation
            self._save_state_locked()
            return escalation.model_copy(deep=True)

    def list_alert_escalations(self, alert_id: str = "") -> list[AlertEscalation]:
        alert_id = alert_id.strip()
        with self._state_lock:
            items = [
                escalation.model_copy(deep=True)
                for escalation in self._state.alert_escalations.values()
                if alert_id == "" or escalation.alert_id == alert_id
            ]
        return sorted(items, key=lambda item: item.created_at)

    def list_aggregation_rules(self) -> list[AggregationRule]:
        with self._state_lock:
            items = [
                rule.model_copy(deep=True) for rule in self._state.aggregation_rules.values()
            ]
        return sorted(items, key=lambda item: (item.priority, item.created_at))

    def upsert_aggregation_rule(self, rule: AggregationRule) -> AggregationRule:
        rule = AggregationRule.model_validate(rule)
        rule.id = rule.id.strip()
        rule.name = rule.name.strip()
        if rule.name == "":
            raise OnCallValidationError("aggregation rule name is required")

        if len(rule.group_by) == 0:
            rule.group_by = ["service", "severity", "name"]
        if rule.window_minutes <= 0:
            rule.window_minutes = 60

        with self._state_lock:
            now = self._utc_now()
            existing = self._state.aggregation_rules.get(rule.id) if rule.id else None
            payload = AggregationRule(
                id=rule.id or new_id("agr"),
                name=rule.name,
                enabled=bool(rule.enabled),
                priority=rule.priority,
                group_by=compact_strings(rule.group_by),
                window_minutes=rule.window_minutes,
                matchers=normalize_matchers(rule.matchers),
                created_at=existing.created_at if existing else now,
                updated_at=now,
            )
            self._state.aggregation_rules[payload.id] = payload
            self._save_state_locked()
            return payload.model_copy(deep=True)

    def delete_aggregation_rule(self, rule_id: str) -> None:
        rule_id = rule_id.strip()
        if rule_id == "":
            raise OnCallValidationError("aggregation rule id is required")
        with self._state_lock:
            if rule_id not in self._state.aggregation_rules:
                raise OnCallNotFound(f"aggregation rule {rule_id}")
            del self._state.aggregation_rules[rule_id]
            self._save_state_locked()

    def list_alert_groups(self) -> list[AlertGroup]:
        with self._state_lock:
            groups = [
                group.model_copy(deep=True)
                for group in self._state.alert_groups.values()
            ]
        return sorted(groups, key=lambda item: item.last_seen, reverse=True)

    def aggregate_alert(self, alert: AlertSignal) -> AggregationResult:
        alert = AlertSignal.model_validate(alert)
        alert = normalize_alert_signal(alert)
        starts_at = ensure_utc(alert.starts_at) or self._utc_now()
        alert.starts_at = starts_at
        if alert.id.strip() == "" and alert.name.strip() == "":
            raise OnCallValidationError("alert id or name is required")

        with self._state_lock:
            rule = self._match_aggregation_rule_locked(alert)
            dedupe_key = self._aggregation_dedupe_key(rule, alert)
            fingerprint = self._alert_fingerprint(alert)
            group_id = self._find_open_group_locked(rule, dedupe_key, starts_at)
            new_group = group_id == ""
            duplicate = False

            if new_group:
                group = self._create_new_group(rule, dedupe_key, alert, starts_at)
            else:
                group = self._state.alert_groups[group_id].model_copy(deep=True)
                duplicate = fingerprint in group.fingerprints
                if not duplicate:
                    alert_identity = first_non_blank(alert.id, fingerprint)
                    if alert_identity and alert_identity not in group.alert_ids:
                        group.alert_ids.append(alert_identity)
                    if fingerprint not in group.fingerprints:
                        group.fingerprints.append(fingerprint)

            if starts_at < group.first_seen:
                group.first_seen = starts_at
            if starts_at > group.last_seen:
                group.last_seen = starts_at
            group.count = len(group.alert_ids)
            group.updated_at = self._utc_now()

            routing = self.route_alert(alert, starts_at)
            if routing.matched:
                group.team = first_non_blank(routing.team, group.team)
                group.owner = first_non_blank(routing.assignee, group.owner)

            self._state.alert_groups[group.id] = group
            self._save_state_locked()

            return AggregationResult(
                group=group,
                new_group=new_group,
                duplicate=duplicate,
                routing=routing,
            )

    def list_routing_rules(self) -> list[RoutingRule]:
        with self._state_lock:
            items = [rule.model_copy(deep=True) for rule in self._state.routing_rules.values()]
        return sorted(items, key=lambda item: (item.priority, item.created_at))

    def upsert_routing_rule(self, rule: RoutingRule) -> RoutingRule:
        return self._save_routing_rule(rule, require_existing=False)

    def update_routing_rule(self, rule: RoutingRule) -> RoutingRule:
        return self._save_routing_rule(rule, require_existing=True)

    def delete_routing_rule(self, rule_id: str) -> None:
        rule_id = rule_id.strip()
        if rule_id == "":
            raise OnCallValidationError("routing rule id is required")
        with self._state_lock:
            if rule_id not in self._state.routing_rules:
                raise OnCallNotFound(f"routing rule {rule_id}")
            del self._state.routing_rules[rule_id]
            self._save_state_locked()

    def route_alert(self, alert: AlertSignal, at: datetime | None = None) -> RoutingDecision:
        alert = AlertSignal.model_validate(alert)
        if at is None:
            at = self._utc_now()
        at = ensure_utc(at) or self._utc_now()

        with self._state_lock:
            alert = normalize_alert_signal(alert)
            return self._route_alert_locked(alert, at)

    def list_schedules(self, team: str = "") -> list[Schedule]:
        team = team.strip()
        with self._state_lock:
            schedules = [
                schedule.model_copy(deep=True)
                for schedule in self._state.schedules.values()
                if team == "" or schedule.team == team
            ]
        return sorted(schedules, key=lambda item: (item.team, item.name))

    def upsert_schedule(self, schedule: Schedule) -> Schedule:
        schedule = Schedule.model_validate(schedule)
        schedule.id = schedule.id.strip()
        schedule.name = schedule.name.strip()
        schedule.team = schedule.team.strip()
        schedule.timezone = schedule.timezone.strip() or "UTC"

        if schedule.name == "":
            raise OnCallValidationError("schedule name is required")
        if schedule.team == "":
            raise OnCallValidationError("schedule team is required")

        try:
            timezone = ZoneInfo(schedule.timezone)
        except (OSError, ValueError) as error:
            raise OnCallValidationError(f"invalid timezone {schedule.timezone}") from error

        if not schedule.layers:
            raise OnCallValidationError("schedule needs at least one layer")

        normalized_layers: list[ScheduleLayer] = []
        for index, layer in enumerate(schedule.layers):
            users = compact_strings(layer.users)
            if not users:
                raise OnCallValidationError("schedule layer users are required")
            start = ensure_utc(layer.start)
            if start is None:
                start = self._utc_now()
            end = ensure_utc(layer.end) if layer.end is not None else None
            rotation_minutes = int(layer.rotation_minutes)
            if rotation_minutes <= 0:
                rotation_minutes = 7 * 24 * 60

            if end is not None and not end > start:
                raise OnCallValidationError("schedule layer end must be after start")

            normalized_layers.append(
                ScheduleLayer(
                    id=layer.id.strip() or new_id("sl"),
                    name=layer.name.strip() or f"Layer {index + 1}",
                    users=users,
                    start=start,
                    end=end,
                    rotation_minutes=rotation_minutes,
                    restrictions=layer.restrictions,
                )
            )

        now = self._utc_now()
        with self._state_lock:
            existing = self._state.schedules.get(schedule.id) if schedule.id else None
            payload = Schedule(
                id=schedule.id or new_id("sch"),
                name=schedule.name,
                team=schedule.team,
                timezone=timezone.key,
                layers=normalized_layers,
                restrictions=schedule.restrictions,
                created_at=existing.created_at if existing else now,
                updated_at=now,
            )
            self._state.schedules[payload.id] = payload
            self._save_state_locked()
            return payload.model_copy(deep=True)

    def delete_schedule(self, schedule_id: str) -> None:
        schedule_id = schedule_id.strip()
        if schedule_id == "":
            raise OnCallValidationError("schedule id is required")

        with self._state_lock:
            if schedule_id not in self._state.schedules:
                raise OnCallNotFound(f"schedule {schedule_id}")
            ref = self._first_routing_rule_for_schedule_locked(schedule_id)
            if ref:
                raise OnCallValidationError(
                    f"schedule {schedule_id} is used by routing rule {ref}"
                )
            del self._state.schedules[schedule_id]
            self._save_state_locked()

    def current_on_call(self, schedule_id: str, at: datetime | None = None) -> OnCallResult:
        if at is None:
            at = self._utc_now()
        schedule_id = schedule_id.strip()
        if schedule_id == "":
            raise OnCallValidationError("schedule id is required")
        at = ensure_utc(at) or self._utc_now()
        with self._state_lock:
            schedule = self._state.schedules.get(schedule_id)
            if schedule is None:
                raise OnCallNotFound(f"schedule {schedule_id}")
            return self._current_on_call_for_schedule_locked(schedule, at)

    def current_on_call_by_team(self, team: str, at: datetime | None = None) -> list[OnCallResult]:
        if at is None:
            at = self._utc_now()
        at = ensure_utc(at) or self._utc_now()
        team = team.strip()
        if team == "":
            raise OnCallValidationError("team is required")

        results: list[OnCallResult] = []
        with self._state_lock:
            for schedule in self._state.schedules.values():
                if schedule.team != team:
                    continue
                result = self._current_on_call_for_schedule_locked(schedule, at)
                results.append(result)
        results.sort(key=lambda result: result.schedule_name)
        return results

    def schedule_shifts(
        self,
        schedule_id: str,
        start: datetime | None,
        end: datetime | None,
        limit: int,
    ) -> list[OnCallShift]:
        schedule_id = schedule_id.strip()
        if schedule_id == "":
            raise OnCallValidationError("schedule id is required")

        with self._state_lock:
            schedule = self._state.schedules.get(schedule_id)
            if schedule is None:
                raise OnCallNotFound(f"schedule {schedule_id}")

            now = self._utc_now()
            start_utc = ensure_utc(start) or now
            if end is None or ensure_utc(end) is None or not ensure_utc(end) > start_utc:
                end_utc = start_utc + timedelta(days=7)
            else:
                end_utc = ensure_utc(end)  # type: ignore[arg-type]

            limit = max(1, min(limit if limit else 100, 500))
            timezone = ZoneInfo(schedule.timezone or "UTC")
            shifts: list[OnCallShift] = []
            for layer in schedule.layers:
                shifts.extend(
                    self._shifts_for_layer(
                        schedule,
                        layer,
                        start_utc,
                        end_utc,
                        timezone,
                        limit,
                    )
                )

            shifts.sort(key=lambda shift: (shift.start, shift.layer_id))
            return shifts[:limit]

    # ---------- Internal helpers ----------

    def _load_state(self) -> None:
        try:
            payload = self._repository.load()
            if isinstance(payload, dict):
                self._state = OnCallState.model_validate(payload)
            else:
                self._state = OnCallState.model_validate({})
        except Exception:
            self._state = OnCallState()

        if self._state.escalation_policies is None:
            self._state.escalation_policies = {}
        if self._state.alert_escalations is None:
            self._state.alert_escalations = {}
        if self._state.aggregation_rules is None:
            self._state.aggregation_rules = {}
        if self._state.alert_groups is None:
            self._state.alert_groups = {}
        if self._state.schedules is None:
            self._state.schedules = {}
        if self._state.routing_rules is None:
            self._state.routing_rules = {}

    def _save_state_locked(self) -> None:
        self._repository.save(self._state.model_dump(mode="json"))

    def _first_routing_rule_for_policy_locked(self, policy_id: str) -> str:
        for rule in self._sorted_routing_rules(self._state.routing_rules.values()):
            if rule.escalation_policy_id == policy_id:
                return rule.id
        return ""

    def _first_routing_rule_for_schedule_locked(self, schedule_id: str) -> str:
        for rule in self._sorted_routing_rules(self._state.routing_rules.values()):
            if rule.schedule_id == schedule_id:
                return rule.id
        return ""

    def _targets_for_escalation_level(self, policy: EscalationPolicy, level: int) -> list[str]:
        for rule in policy.rules:
            if rule.level == level:
                return compact_strings(rule.notify_targets)
        return []

    def _next_escalation_level_locked(self, alert_id: str) -> int:
        max_level = 0
        for escalation in self._state.alert_escalations.values():
            if escalation.alert_id == alert_id and escalation.level > max_level:
                max_level = escalation.level
        return max_level + 1

    def _match_aggregation_rule_locked(self, alert: AlertSignal) -> AggregationRule:
        candidates = [
            rule.model_copy(deep=True)
            for rule in self._state.aggregation_rules.values()
            if rule.enabled and match_all(rule.matchers, alert)
        ]
        if candidates:
            candidates.sort(key=lambda rule: (rule.priority, rule.created_at))
            chosen = candidates[0]
            if not chosen.group_by:
                chosen.group_by = ["service", "severity", "name"]
            if chosen.window_minutes <= 0:
                chosen.window_minutes = 60
            return chosen
        return AggregationRule(
            id="default",
            name="Default aggregation",
            enabled=True,
            group_by=["service", "severity", "name"],
            window_minutes=60,
        )

    def _find_open_group_locked(
        self,
        rule: AggregationRule,
        dedupe_key: str,
        at: datetime,
    ) -> str:
        window = timedelta(minutes=rule.window_minutes)
        if window <= timedelta(0):
            window = timedelta(hours=1)

        candidates: list[tuple[str, datetime]] = []
        for group in self._state.alert_groups.values():
            if (
                group.status != GROUP_STATUS_OPEN
                or group.rule_id != rule.id
                or group.dedupe_key != dedupe_key
            ):
                continue
            delta = at - group.last_seen
            if delta.total_seconds() < 0:
                delta = -delta
            if delta <= window:
                candidates.append((group.id, group.last_seen))

        candidates.sort(key=lambda item: item[1], reverse=True)
        return candidates[0][0] if candidates else ""

    def _create_new_group(
        self,
        rule: AggregationRule,
        dedupe_key: str,
        alert: AlertSignal,
        starts_at: datetime,
    ) -> AlertGroup:
        alert_id = first_non_blank(alert.id, self._alert_fingerprint(alert))
        return AlertGroup(
            id=stable_id(
                "grp",
                f"{rule.id}:{dedupe_key}:{int(starts_at.timestamp() * 1_000_000_000)}",
            ),
            rule_id=rule.id,
            dedupe_key=dedupe_key,
            name=first_non_blank(alert.name, "Alert group"),
            alert_ids=[alert_id],
            fingerprints=[self._alert_fingerprint(alert)],
            labels=dict(alert.labels),
            service=alert.service,
            severity=alert.severity,
            team=alert.team,
            owner="",
            status=GROUP_STATUS_OPEN,
            count=1,
            first_seen=starts_at,
            last_seen=starts_at,
            created_at=self._utc_now(),
            updated_at=self._utc_now(),
        )

    def _aggregation_dedupe_key(self, rule: AggregationRule, alert: AlertSignal) -> str:
        parts = [f"rule={rule.id}"]
        for field in rule.group_by:
            normalized_field = field.strip()
            if not normalized_field:
                continue
            value, exists = alert_field_value(alert, normalized_field)
            if not exists:
                value = "unknown"
            parts.append(f"{normalized_field.lower()}={value}")
        return "|".join(parts)

    def _alert_fingerprint(self, alert: AlertSignal) -> str:
        if alert.id.strip():
            return f"id:{alert.id.strip()}"
        values = [
            f"name={alert.name}",
            f"service={alert.service}",
            f"severity={alert.severity}",
            f"source={alert.source}",
        ]
        for key in sorted(alert.labels.keys()):
            values.append(f"label.{key}={alert.labels[key]}")
        fingerprint = hashlib.sha1("|".join(values).encode("utf-8")).hexdigest()
        return f"fp:{fingerprint}"

    def _sorted_routing_rules(self, rules: Any) -> list[RoutingRule]:
        sorted_rules = [rule.model_copy(deep=True) for rule in rules]
        sorted_rules.sort(key=lambda item: (item.priority, item.created_at))
        return sorted_rules

    def _route_alert_locked(self, alert: AlertSignal, at: datetime) -> RoutingDecision:
        for rule in self._sorted_routing_rules(self._state.routing_rules.values()):
            if not self._match_routing_rule(rule, alert):
                continue
            decision = RoutingDecision(
                matched=True,
                rule_id=rule.id,
                rule_name=rule.name,
                team=first_non_blank(rule.team, alert.team),
                schedule_id=rule.schedule_id,
                escalation_policy_id=rule.escalation_policy_id,
                matched_at=at,
            )
            if rule.auto_assign:
                decision.assignee = self._auto_assign_locked(rule, decision.team, at)
            return decision
        return RoutingDecision(matched=False, matched_at=at)

    def _match_routing_rule(self, rule: RoutingRule, alert: AlertSignal) -> bool:
        if not rule.enabled:
            return False
        return match_all(rule.matchers, alert)

    def _save_routing_rule(self, rule: RoutingRule, require_existing: bool) -> RoutingRule:
        rule = RoutingRule.model_validate(rule)
        rule.id = rule.id.strip()
        rule.name = rule.name.strip()
        rule.team = rule.team.strip()
        rule.schedule_id = rule.schedule_id.strip()
        rule.escalation_policy_id = rule.escalation_policy_id.strip()

        if rule.name == "":
            raise OnCallValidationError("routing rule name is required")
        if require_existing and not rule.id:
            raise OnCallValidationError("routing rule id is required")
        if not (rule.team or rule.schedule_id or rule.escalation_policy_id):
            raise OnCallValidationError("routing target is required")

        with self._state_lock:
            if rule.schedule_id and rule.schedule_id not in self._state.schedules:
                raise OnCallNotFound(f"schedule {rule.schedule_id}")
            if rule.escalation_policy_id and rule.escalation_policy_id not in self._state.escalation_policies:
                raise OnCallNotFound(f"escalation policy {rule.escalation_policy_id}")

            existing = self._state.routing_rules.get(rule.id)
            if require_existing and existing is None:
                raise OnCallNotFound(f"routing rule {rule.id}")

            now = self._utc_now()
            payload = RoutingRule(
                id=rule.id or new_id("rr"),
                name=rule.name,
                enabled=rule.enabled,
                priority=rule.priority,
                team=rule.team,
                schedule_id=rule.schedule_id,
                escalation_policy_id=rule.escalation_policy_id,
                auto_assign=rule.auto_assign,
                matchers=normalize_matchers(rule.matchers),
                created_at=(existing.created_at if existing else now),
                updated_at=now,
            )
            self._state.routing_rules[payload.id] = payload
            self._save_state_locked()
            return payload.model_copy(deep=True)

    def _auto_assign_locked(self, rule: RoutingRule, team: str, at: datetime) -> str:
        if rule.schedule_id:
            schedule = self._state.schedules.get(rule.schedule_id)
            if schedule is not None:
                result = self._current_on_call_for_schedule_locked(schedule, at)
                if result.users:
                    return result.users[0]

        for schedule in sorted(
            self._state.schedules.values(),
            key=lambda schedule: (schedule.team, schedule.name),
        ):
            if team and schedule.team != team:
                continue
            result = self._current_on_call_for_schedule_locked(schedule, at)
            if result.users:
                return result.users[0]
        return ""

    def _current_on_call_for_schedule_locked(
        self,
        schedule: Schedule,
        at_utc: datetime,
    ) -> OnCallResult:
        loc = ZoneInfo(schedule.timezone or "UTC")
        local_at = at_utc.astimezone(loc)
        result = OnCallResult(
            schedule_id=schedule.id,
            schedule_name=schedule.name,
            team=schedule.team,
            timezone=loc.key,
            at=at_utc,
            users=[],
            shifts=[],
        )
        if not self._matches_time_restrictions(schedule.restrictions, local_at):
            return result

        seen_users: set[str] = set()
        for layer in schedule.layers:
            if not self._matches_time_restrictions(layer.restrictions, local_at):
                continue
            shift = self._current_shift_for_layer(schedule, layer, at_utc, loc)
            if shift is None:
                continue
            if shift.user not in seen_users:
                seen_users.add(shift.user)
                result.users.append(shift.user)
            result.shifts.append(shift)
        return result

    def _current_shift_for_layer(
        self,
        schedule: Schedule,
        layer: ScheduleLayer,
        at_utc: datetime,
        loc: ZoneInfo,
    ) -> OnCallShift | None:
        users = compact_strings(layer.users)
        if not users:
            return None

        rotation = timedelta(minutes=layer.rotation_minutes)
        if rotation <= timedelta(0):
            rotation = timedelta(days=7)

        local_at = at_utc.astimezone(loc)
        start = ensure_utc(layer.start)
        if start is None:
            return None
        layer_start = start.astimezone(loc)
        if local_at < layer_start:
            return None

        layer_end = ensure_utc(layer.end)
        if layer_end is not None and not local_at < layer_end.astimezone(loc):
            return None

        elapsed = local_at - layer_start
        bucket = int(elapsed.total_seconds() // rotation.total_seconds())
        if bucket < 0:
            return None

        user = users[bucket % len(users)]
        shift_start = layer_start + timedelta(seconds=bucket * rotation.total_seconds())
        shift_end = shift_start + rotation
        if layer_end is not None:
            layer_end_local = layer_end.astimezone(loc)
            if shift_end > layer_end_local:
                shift_end = layer_end_local

        return OnCallShift(
            id=stable_id(
                "shift",
                f"{schedule.id}:{layer.id}:{user}:{int(shift_start.timestamp())}",
            ),
            schedule_id=schedule.id,
            schedule_name=schedule.name,
            team=schedule.team,
            layer_id=layer.id,
            layer_name=layer.name,
            user=user,
            start=shift_start.astimezone(UTC),
            end=shift_end.astimezone(UTC),
            rotation_minutes=int(rotation.total_seconds() / 60),
        )

    def _shifts_for_layer(
        self,
        schedule: Schedule,
        layer: ScheduleLayer,
        start: datetime,
        end: datetime,
        loc: ZoneInfo,
        limit: int,
    ) -> list[OnCallShift]:
        users = compact_strings(layer.users)
        if not users:
            return []

        rotation = timedelta(minutes=layer.rotation_minutes)
        if rotation <= timedelta(0):
            rotation = timedelta(days=7)

        layer_start = ensure_utc(layer.start)
        if layer_start is None:
            return []
        layer_start_local = layer_start.astimezone(loc)

        window_start = start.astimezone(loc)
        window_end = end.astimezone(loc)
        layer_end = ensure_utc(layer.end)
        if layer_end is not None:
            layer_end_local = layer_end.astimezone(loc)
            if layer_end_local < window_end:
                window_end = layer_end_local

        if not window_end > layer_start_local:
            return []

        if window_start > layer_start_local:
            elapsed = window_start - layer_start_local
            bucket = int(elapsed.total_seconds() // rotation.total_seconds())
            shift_start = layer_start_local + timedelta(seconds=bucket * rotation.total_seconds())
        else:
            shift_start = layer_start_local

        out: list[OnCallShift] = []
        while shift_start < window_end and len(out) < limit:
            shift_end = shift_start + rotation
            if layer_end is not None:
                layer_end_local = layer_end.astimezone(loc)
                if shift_end > layer_end_local:
                    shift_end = layer_end_local

            overlap_start = max_time(shift_start, window_start)
            overlap_end = min_time(shift_end, window_end)

            if self._shift_matches_restrictions(
                schedule,
                layer,
                overlap_start,
                overlap_end,
                loc,
            ):
                shift = self._current_shift_for_layer(schedule, layer, overlap_start.astimezone(UTC), loc)
                if shift is not None and shift.end > start and shift.start < end:
                    out.append(shift)

            shift_start = shift_start + rotation
        return out

    def _shift_matches_restrictions(
        self,
        schedule: Schedule,
        layer: ScheduleLayer,
        local_start: datetime,
        local_end: datetime,
        loc: ZoneInfo,
    ) -> bool:
        if not local_end > local_start:
            return False
        for probe in self._restriction_probe_times(
            local_start,
            local_end,
            loc,
            schedule.restrictions,
            layer.restrictions,
        ):
            if probe < local_start or not probe < local_end:
                continue
            if (
                self._matches_time_restrictions(schedule.restrictions, probe)
                and self._matches_time_restrictions(layer.restrictions, probe)
            ):
                return True
        return False

    def _restriction_probe_times(
        self,
        local_start: datetime,
        local_end: datetime,
        loc: ZoneInfo,
        *restriction_sets: list[TimeRestriction],
    ) -> list[datetime]:
        probes: list[datetime] = [local_start]
        start_day = datetime(
            local_start.year,
            local_start.month,
            local_start.day,
            0,
            0,
            0,
            tzinfo=loc,
        ) - timedelta(days=1)
        end_day = datetime(
            local_end.year,
            local_end.month,
            local_end.day,
            0,
            0,
            0,
            tzinfo=loc,
        ) + timedelta(days=1)

        day = start_day
        while day <= end_day:
            for restrictions in restriction_sets:
                for restriction in restrictions:
                    probes.append(
                        day + timedelta(minutes=normalize_start_minute(restriction.start_minute))
                    )
            day += timedelta(days=1)

        probes = sorted(set(probes))
        return probes

    def _matches_time_restrictions(self, restrictions: list[TimeRestriction], local: datetime) -> bool:
        if not restrictions:
            return True
        for restriction in restrictions:
            if self._restriction_matches(restriction, local):
                return True
        return False

    def _restriction_matches(self, restriction: TimeRestriction, local: datetime) -> bool:
        start = normalize_start_minute(restriction.start_minute)
        end = normalize_end_minute(restriction.end_minute)
        minute = local.hour * 60 + local.minute

        if start == end:
            return weekday_allowed(restriction.weekdays, local.weekday())
        if start < end:
            return (
                weekday_allowed(restriction.weekdays, local.weekday())
                and start <= minute < end
            )
        if minute >= start:
            return weekday_allowed(restriction.weekdays, local.weekday())
        if minute < end:
            return weekday_allowed(restriction.weekdays, (local - timedelta(days=1)).weekday())
        return False

    @staticmethod
    def _utc_now() -> datetime:
        return datetime.now(timezone.utc)


def _default_state_file() -> Path:
    path = os.getenv("SUPER_BIZ_AGENT_ONCALL_STATE_FILE")
    if path:
        return Path(path)
    temp_dir = os.environ.get("TMPDIR")
    if temp_dir and temp_dir.strip():
        base = Path(temp_dir)
    else:
        base = Path("/tmp")
    return base.joinpath("super_biz_agent", "oncall_state.json")
