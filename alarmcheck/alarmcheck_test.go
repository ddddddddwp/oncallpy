package main

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestParseFaultCodes(t *testing.T) {
	got := parseFaultCodes("010a, 010B;010A\n010C")
	want := []string{"010A", "010B", "010C"}
	if len(got) != len(want) {
		t.Fatalf("parseFaultCodes length = %d, want %d: %#v", len(got), len(want), got)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("parseFaultCodes[%d] = %s, want %s", i, got[i], want[i])
		}
	}
}

func TestAPIURL(t *testing.T) {
	tests := map[string]string{
		"http://127.0.0.1:6872":     "http://127.0.0.1:6872/api/alerts/ingest",
		"http://127.0.0.1:6872/":    "http://127.0.0.1:6872/api/alerts/ingest",
		"http://127.0.0.1:6872/api": "http://127.0.0.1:6872/api/alerts/ingest",
	}
	for base, want := range tests {
		if got := apiURL(base, "/alerts/ingest"); got != want {
			t.Fatalf("apiURL(%q) = %q, want %q", base, got, want)
		}
	}
}

func TestDoJSONEnvelope(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"message":"OK","data":{"alertId":"a1","action":"created"}}`))
	}))
	defer server.Close()

	var response ingestResponse
	if err := getJSON(t.Context(), server.Client(), server.URL, &response); err != nil {
		t.Fatalf("getJSON() error = %v", err)
	}
	if response.AlertID != "a1" || response.Action != "created" {
		t.Fatalf("response = %+v", response)
	}
}

func TestBuildPayloadDoesNotContainSolution(t *testing.T) {
	payload := buildPayload(profileForFault("010A"), 1, 1)
	if _, ok := payload.Alert.Attributes["solutionCode"]; ok {
		t.Fatal("alert attributes must not contain solutionCode")
	}
	if _, ok := payload.Raw["solutionCode"]; ok {
		t.Fatal("raw payload must not contain solutionCode")
	}
	if payload.Alert.FaultCode != "010A" {
		t.Fatalf("faultCode = %s", payload.Alert.FaultCode)
	}
}

func TestUnknownFaultProfileDerivesExpectedSolution(t *testing.T) {
	profile := profileForFault("1159")
	if profile.SolutionCode != 3 {
		t.Fatalf("solutionCode = %d, want 3", profile.SolutionCode)
	}
	if profile.Command != "observe_only" {
		t.Fatalf("command = %s, want observe_only", profile.Command)
	}
	if profile.ExpectedState != "investigating" {
		t.Fatalf("expectedState = %s, want investigating", profile.ExpectedState)
	}
}

func TestKnownExtendedFaultProfile(t *testing.T) {
	profile := profileForFault("012A")
	if profile.Name != "电网包络异常" {
		t.Fatalf("name = %s", profile.Name)
	}
	if profile.AlarmLevel != "1级-低级" {
		t.Fatalf("alarmLevel = %s", profile.AlarmLevel)
	}
	if profile.SolutionCode != 4 || profile.Command != "reset_after_battery_check" {
		t.Fatalf("solution = %d command = %s", profile.SolutionCode, profile.Command)
	}
}

func TestMockCloudFixedServerRecordsCommands(t *testing.T) {
	cloud, err := newMockCloud("127.0.0.1:0", "/api/cloud/commands")
	if err != nil {
		t.Fatalf("newMockCloud() error = %v", err)
	}
	defer cloud.Close()

	body := []byte(`{"commandId":"cmd-1","alertId":"alert-1","deviceId":"dev-1","faultCode":"010A","command":"restart_inverter","operator":"test","payload":{"solutionCode":0}}`)
	resp, err := http.Post(cloud.URL(), "application/json", bytes.NewReader(body))
	if err != nil {
		t.Fatalf("post command error = %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d", resp.StatusCode)
	}

	command, ok := cloud.findByAlertID("alert-1")
	if !ok {
		t.Fatal("findByAlertID did not find alert-1")
	}
	if command.Command != "restart_inverter" || command.FaultCode != "010A" {
		t.Fatalf("command = %+v", command)
	}

	apiResp, err := http.Get(cloud.WebURL() + "api/commands")
	if err != nil {
		t.Fatalf("get api commands error = %v", err)
	}
	defer apiResp.Body.Close()
	var payload struct {
		Count   int                  `json:"count"`
		Records []cloudCommandRecord `json:"records"`
	}
	if err := json.NewDecoder(apiResp.Body).Decode(&payload); err != nil {
		t.Fatalf("decode api commands error = %v", err)
	}
	if payload.Count != 1 || len(payload.Records) != 1 {
		t.Fatalf("payload = %+v", payload)
	}
}
