package main

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"
)

func TestBuildPayloadUsesConfiguredFaultCode(t *testing.T) {
	cfg := config{
		FaultCode: "E_TEMP_001",
		Severity:  "critical",
		Source:    "device-cloud",
		Mode:      "firing",
		Seed:      42,
		SameAlert: true,
	}

	payload := buildPayload(cfg, 7)
	if payload.Alert.FaultCode != "E_TEMP_001" {
		t.Fatalf("fault code = %q, want E_TEMP_001", payload.Alert.FaultCode)
	}
	if payload.SourceAlertID != "ESS-SH-001:E_TEMP_001" {
		t.Fatalf("source alert id = %q, want ESS-SH-001:E_TEMP_001", payload.SourceAlertID)
	}
	if payload.Alert.DeviceID == "" || payload.Alert.MetricName == "" {
		t.Fatalf("payload missing realistic device or metric fields: %+v", payload.Alert)
	}
}

func TestBuildPayloadUsesHaisoFaultCatalogWithoutSolutionFields(t *testing.T) {
	cfg := config{
		FaultCode: "010A",
		Severity:  "critical",
		Source:    "device-cloud",
		Mode:      "firing",
		Seed:      42,
		SameAlert: true,
	}

	payload := buildPayload(cfg, 1)
	if payload.Alert.FaultCode != "010A" {
		t.Fatalf("fault code = %q, want 010A", payload.Alert.FaultCode)
	}
	if payload.Alert.FaultName != "dcdc1 温度采样错误" {
		t.Fatalf("fault name = %q, want dcdc1 温度采样错误", payload.Alert.FaultName)
	}
	if payload.SourceAlertID != "HS-INV-SH-001:010A" {
		t.Fatalf("source alert id = %q, want HS-INV-SH-001:010A", payload.SourceAlertID)
	}
	if payload.Alert.ProductKey != "HS-ESS25K" || payload.Alert.DeviceID == "" {
		t.Fatalf("payload missing haiso device identity: %+v", payload.Alert)
	}
	if _, ok := payload.Alert.Labels["mock_solution_code"]; ok {
		t.Fatalf("alert labels must not include solution code: %+v", payload.Alert.Labels)
	}
	if _, ok := payload.Alert.Labels["mock_command"]; ok {
		t.Fatalf("alert labels must not include mock command: %+v", payload.Alert.Labels)
	}
	if _, ok := payload.Alert.Annotations["mock_solution_method"]; ok {
		t.Fatalf("alert annotations must not include solution method: %+v", payload.Alert.Annotations)
	}
	if _, ok := payload.Alert.Annotations["mock_command"]; ok {
		t.Fatalf("alert annotations must not include mock command: %+v", payload.Alert.Annotations)
	}
	if _, ok := payload.Alert.Attributes["mockSolutionCode"]; ok {
		t.Fatalf("alert attributes must not include solution code: %+v", payload.Alert.Attributes)
	}
	if _, ok := payload.Alert.Attributes["mockCommand"]; ok {
		t.Fatalf("alert attributes must not include mock command: %+v", payload.Alert.Attributes)
	}
	if _, ok := payload.Raw["mockSolutionCode"]; ok {
		t.Fatalf("raw payload must not include solution code: %+v", payload.Raw)
	}
	if _, ok := payload.Raw["mockCommand"]; ok {
		t.Fatalf("raw payload must not include mock command: %+v", payload.Raw)
	}
}

func TestSelectFaultCodeSupportsCommaList(t *testing.T) {
	cfg := config{
		FaultCode: "010A,0116,1159",
		Severity:  "warning",
		Source:    "device-cloud",
		Mode:      "firing",
		Seed:      1,
	}

	seen := map[string]bool{}
	for i := 1; i <= 20; i++ {
		payload := buildPayload(cfg, i)
		seen[payload.Alert.FaultCode] = true
		switch payload.Alert.FaultCode {
		case "010A", "0116", "1159":
		default:
			t.Fatalf("unexpected fault code %q", payload.Alert.FaultCode)
		}
	}
	if len(seen) < 2 {
		t.Fatalf("comma list did not vary fault codes enough: %+v", seen)
	}
}

func TestExtractActionFromWrappedResponse(t *testing.T) {
	action := extractAction([]byte(`{"code":0,"data":{"action":"created"}}`))
	if action != "created" {
		t.Fatalf("action = %q, want created", action)
	}
}

func TestRunSendsRequestedCount(t *testing.T) {
	var received atomic.Int64
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			t.Fatalf("method = %s, want POST", r.Method)
		}
		defer r.Body.Close()

		var payload ingestPayload
		if err := json.NewDecoder(r.Body).Decode(&payload); err != nil {
			t.Fatalf("Decode() error = %v", err)
		}
		if payload.Alert.FaultCode != "E_NET_LOST" {
			t.Fatalf("fault code = %q, want E_NET_LOST", payload.Alert.FaultCode)
		}
		received.Add(1)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"code":0,"data":{"action":"created"}}`))
	}))
	defer server.Close()

	cfg := config{
		URL:         server.URL,
		Count:       12,
		Concurrency: 4,
		FaultCode:   "E_NET_LOST",
		Severity:    "warning",
		Source:      "device-cloud",
		Mode:        "firing",
		Timeout:     2 * time.Second,
		Seed:        100,
	}

	summary := run(context.Background(), cfg, &http.Client{Timeout: cfg.Timeout})
	if summary.total != cfg.Count || summary.success != cfg.Count || summary.failed != 0 {
		t.Fatalf("summary = total:%d success:%d failed:%d", summary.total, summary.success, summary.failed)
	}
	if received.Load() != int64(cfg.Count) {
		t.Fatalf("received = %d, want %d", received.Load(), cfg.Count)
	}
	if summary.actions["created"] != cfg.Count {
		t.Fatalf("created action count = %d, want %d", summary.actions["created"], cfg.Count)
	}
}
