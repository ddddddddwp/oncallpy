from datetime import timezone

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api import oncall
from app.domains.oncall import OnCallStore
from app.domains.oncall.utils import now_utc


def _client(tmp_path, monkeypatch) -> TestClient:
    monkeypatch.setattr(
        oncall,
        "_store",
        OnCallStore(state_file=tmp_path / "oncall-state.json"),
    )
    application = FastAPI()
    application.include_router(oncall.router, prefix="/api")
    return TestClient(application)


def test_alert_contract_accepts_source_field_aliases() -> None:
    from app.domains.alerts.models import ExternalAlertPayload

    payload = ExternalAlertPayload.model_validate(
        {
            "faultCode": "E_VOLT_HIGH",
            "faultName": "英威腾逆变器电压过高",
            "faultCategory": "electrical",
        }
    )

    assert payload.fault_code == "E_VOLT_HIGH"
    assert payload.fault_name == "英威腾逆变器电压过高"
    assert payload.fault_category == "electrical"


def test_oncall_utc_helper_returns_aware_utc_time() -> None:
    assert now_utc().tzinfo == timezone.utc


def test_alert_group_endpoint_aggregates_once(tmp_path, monkeypatch) -> None:
    client = _client(tmp_path, monkeypatch)

    response = client.post(
        "/api/alert-groups",
        json={
            "id": "alert-1",
            "name": "HighVoltage",
            "service": "inverter",
            "severity": "critical",
        },
    )

    assert response.status_code == 200
    result = response.json()["result"]
    assert result["newGroup"] is True
    assert result["duplicate"] is False
    assert result["group"]["count"] == 1


def test_escalation_policy_endpoint_persists_state(tmp_path, monkeypatch) -> None:
    client = _client(tmp_path, monkeypatch)

    created = client.post(
        "/api/escalation-policies",
        json={"name": "primary", "team": "energy"},
    )
    listed = client.get("/api/escalation-policies")

    assert created.status_code == 200
    assert listed.status_code == 200
    assert listed.json()["policies"][0]["name"] == "primary"
