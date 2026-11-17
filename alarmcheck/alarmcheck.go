package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"html/template"
	"io"
	"math/rand"
	"net"
	"net/http"
	"net/url"
	"os"
	"os/signal"
	"sort"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"time"
)

type config struct {
	OncallURL      string
	FaultCodes     string
	Timeout        time.Duration
	Delay          time.Duration
	Operator       string
	TimeoutSeconds int
	Seed           int64
	CloudAddr      string
	CloudPath      string
	ServeOnly      bool
	KeepServer     bool
}

type apiEnvelope struct {
	Message string          `json:"message"`
	Data    json.RawMessage `json:"data"`
}

type ingestPayload struct {
	Source        string                 `json:"source"`
	SourceEventID string                 `json:"sourceEventId"`
	SourceAlertID string                 `json:"sourceAlertId"`
	EventType     string                 `json:"eventType"`
	Actor         string                 `json:"actor"`
	OccurredAt    time.Time              `json:"occurredAt"`
	Alert         externalAlertPayload   `json:"alert"`
	Raw           map[string]interface{} `json:"raw,omitempty"`
}

type externalAlertPayload struct {
	Name          string                 `json:"name"`
	FaultCode     string                 `json:"faultCode"`
	FaultName     string                 `json:"faultName"`
	FaultCategory string                 `json:"faultCategory"`
	Severity      string                 `json:"severity"`
	Service       string                 `json:"service"`
	Team          string                 `json:"team"`
	Region        string                 `json:"region"`
	Environment   string                 `json:"environment"`
	DeviceID      string                 `json:"deviceId"`
	ProductKey    string                 `json:"productKey"`
	GatewayID     string                 `json:"gatewayId"`
	MetricName    string                 `json:"metricName"`
	MetricValue   string                 `json:"metricValue"`
	Threshold     string                 `json:"threshold"`
	Unit          string                 `json:"unit"`
	Description   string                 `json:"description"`
	StartsAt      time.Time              `json:"startsAt"`
	Labels        map[string]string      `json:"labels,omitempty"`
	Annotations   map[string]string      `json:"annotations,omitempty"`
	Attributes    map[string]interface{} `json:"attributes,omitempty"`
}

type ingestResponse struct {
	IngestID string `json:"ingestId"`
	EventID  string `json:"eventId"`
	AlertID  string `json:"alertId"`
	Action   string `json:"action"`
	Message  string `json:"message"`
}

type remediateRequest struct {
	CloudURL       string                 `json:"cloudUrl"`
	Operator       string                 `json:"operator"`
	TimeoutSeconds int                    `json:"timeoutSeconds"`
	Payload        map[string]interface{} `json:"payload,omitempty"`
}

type remediateResponse struct {
	Result remediateResult `json:"result"`
}

type remediateResult struct {
	Alert          alert             `json:"alert"`
	AlertID        string            `json:"alertId"`
	DeviceID       string            `json:"deviceId"`
	FaultCode      string            `json:"faultCode"`
	SolutionCode   int               `json:"solutionCode"`
	SolutionMethod string            `json:"solutionMethod"`
	Command        string            `json:"command"`
	CommandID      string            `json:"commandId"`
	CommandResult  cloudSendResult   `json:"commandResult"`
	FinalState     string            `json:"finalState"`
	DryRun         bool              `json:"dryRun"`
	Raw            json.RawMessage   `json:"-"`
	Extra          map[string]string `json:"-"`
}

type cloudSendResult struct {
	CommandID    string `json:"commandId"`
	Accepted     bool   `json:"accepted"`
	StatusCode   int    `json:"statusCode"`
	CloudCode    string `json:"cloudCode"`
	CloudTraceID string `json:"cloudTraceId"`
	CheckSummary string `json:"checkSummary"`
}

type alertDetailResponse struct {
	Alert alert `json:"alert"`
}

type alert struct {
	ID          string `json:"id"`
	Name        string `json:"name"`
	FaultCode   string `json:"faultCode"`
	FaultName   string `json:"faultName"`
	Service     string `json:"service"`
	Severity    string `json:"severity"`
	DeviceID    string `json:"deviceId"`
	State       string `json:"state"`
	Owner       string `json:"owner"`
	Source      string `json:"source"`
	SourceState string `json:"sourceState"`
}

type faultProfile struct {
	Code          string
	Name          string
	EnglishName   string
	Category      string
	MetricName    string
	MetricValue   string
	Threshold     string
	Unit          string
	Severity      string
	AlarmLevel    string
	FaultLevel    int
	SolutionCode  int
	Command       string
	ExpectedState string
}

type cloudCommand struct {
	CommandID string                 `json:"commandId"`
	AlertID   string                 `json:"alertId"`
	DeviceID  string                 `json:"deviceId"`
	FaultCode string                 `json:"faultCode"`
	Command   string                 `json:"command"`
	Operator  string                 `json:"operator"`
	Reason    string                 `json:"reason"`
	Payload   map[string]interface{} `json:"payload"`
}

type cloudCommandRecord struct {
	Index      int             `json:"index"`
	ReceivedAt time.Time       `json:"receivedAt"`
	RemoteAddr string          `json:"remoteAddr"`
	UserAgent  string          `json:"userAgent"`
	Command    cloudCommand    `json:"command"`
	Raw        json.RawMessage `json:"raw"`
}

type mockCloud struct {
	addr        string
	commandPath string
	baseURL     string
	server      *http.Server
	listener    net.Listener
	mu          sync.Mutex
	records     []cloudCommandRecord
}

type checkRecord struct {
	FaultCode     string
	AlertID       string
	DeviceID      string
	SolutionCode  int
	Command       string
	CloudAccepted bool
	FinalState    string
	CloudTraceID  string
}

var mockCloudTemplate = template.Must(template.New("mock-cloud").Funcs(template.FuncMap{
	"pretty": prettyJSON,
}).Parse(`<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>alarmcheck 模拟云端</title>
  <style>
    :root { color-scheme: light; --border:#d7dee8; --muted:#65758b; --bg:#f6f8fb; --panel:#fff; --text:#162033; --accent:#1264d8; }
    * { box-sizing: border-box; }
    body { margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: var(--text); background: var(--bg); }
    header { padding: 20px 28px 14px; background: var(--panel); border-bottom: 1px solid var(--border); }
    h1 { margin: 0 0 8px; font-size: 22px; font-weight: 700; }
    .sub { color: var(--muted); font-size: 14px; line-height: 1.6; }
    main { padding: 20px 28px 36px; }
    .toolbar { display:flex; gap: 12px; align-items:center; justify-content:space-between; margin-bottom: 14px; flex-wrap: wrap; }
    .count { font-weight: 700; }
    button { border: 1px solid var(--border); background: var(--panel); color: var(--text); padding: 8px 12px; border-radius: 6px; cursor: pointer; }
    button.primary { background: var(--accent); color: #fff; border-color: var(--accent); }
    table { width: 100%; border-collapse: collapse; background: var(--panel); border: 1px solid var(--border); }
    th, td { padding: 10px 12px; border-bottom: 1px solid var(--border); text-align: left; vertical-align: top; font-size: 13px; }
    th { background: #eef3fa; font-size: 12px; color: #3a4b63; white-space: nowrap; }
    tr:last-child td { border-bottom: 0; }
    code { background: #eef3fa; border: 1px solid var(--border); padding: 2px 5px; border-radius: 4px; }
    details { min-width: 260px; }
    summary { cursor: pointer; color: var(--accent); }
    pre { margin: 8px 0 0; padding: 10px; max-height: 260px; overflow: auto; background: #0f172a; color: #e5edf7; border-radius: 6px; font-size: 12px; line-height: 1.45; }
    .empty { padding: 24px; background: var(--panel); border: 1px solid var(--border); color: var(--muted); }
    @media (max-width: 900px) {
      main, header { padding-left: 14px; padding-right: 14px; }
      table, thead, tbody, tr, th, td { display:block; }
      thead { display:none; }
      tr { border-bottom: 1px solid var(--border); }
      td { border: 0; padding: 8px 10px; }
      td::before { content: attr(data-label); display:block; color: var(--muted); font-size: 12px; margin-bottom: 3px; }
    }
  </style>
</head>
<body>
  <header>
    <h1>alarmcheck 模拟云端</h1>
    <div class="sub">接收 oncall 自动处置接口发出的运维指令。固定接收地址：<code>{{.CommandEndpoint}}</code></div>
  </header>
  <main>
    <div class="toolbar">
      <div class="count">已接收 {{.Count}} 条指令</div>
      <div>
        <button onclick="location.reload()">刷新</button>
        <button class="primary" onclick="clearRecords()">清空记录</button>
      </div>
    </div>
    {{if .Records}}
    <table>
      <thead>
        <tr>
          <th>#</th>
          <th>接收时间</th>
          <th>故障码</th>
          <th>设备</th>
          <th>告警</th>
          <th>指令</th>
          <th>操作人</th>
          <th>详情</th>
        </tr>
      </thead>
      <tbody>
        {{range .Records}}
        <tr>
          <td data-label="#">{{.Index}}</td>
          <td data-label="接收时间">{{.ReceivedAt.Format "2006-01-02 15:04:05"}}</td>
          <td data-label="故障码"><code>{{.Command.FaultCode}}</code></td>
          <td data-label="设备">{{.Command.DeviceID}}</td>
          <td data-label="告警">{{.Command.AlertID}}</td>
          <td data-label="指令"><code>{{.Command.Command}}</code></td>
          <td data-label="操作人">{{.Command.Operator}}</td>
          <td data-label="详情">
            <details>
              <summary>查看 payload / 原始请求</summary>
              <pre>{{pretty .Command}}</pre>
            </details>
          </td>
        </tr>
        {{end}}
      </tbody>
    </table>
    {{else}}
    <div class="empty">还没有收到运维指令。运行 alarmcheck 验收，或让 oncall 使用上面的固定接收地址发送指令。</div>
    {{end}}
  </main>
  <script>
    async function clearRecords() {
      if (!confirm('确认清空模拟云端收到的所有指令记录？')) return;
      await fetch('/api/commands', { method: 'DELETE' });
      location.reload();
    }
    setInterval(() => location.reload(), 10000);
  </script>
</body>
</html>`))

func main() {
	cfg := parseFlags()

	cloud, err := newMockCloud(cfg.CloudAddr, cfg.CloudPath)
	if err != nil {
		fmt.Fprintln(os.Stderr, "start mock cloud failed:", err)
		os.Exit(1)
	}
	defer cloud.Close()
	fmt.Printf("mock cloud web: %s\n", cloud.WebURL())
	fmt.Printf("mock cloud command endpoint: %s\n", cloud.URL())

	if cfg.ServeOnly {
		fmt.Println("serve-only mode is running, press Ctrl+C to stop")
		waitForInterrupt()
		return
	}

	ctx, cancel := context.WithTimeout(context.Background(), cfg.Timeout)
	defer cancel()

	codes := parseFaultCodes(cfg.FaultCodes)
	if len(codes) == 0 {
		fmt.Fprintln(os.Stderr, "-fault-codes is required")
		os.Exit(2)
	}

	client := &http.Client{Timeout: cfg.Timeout}
	records, err := runCheck(ctx, cfg, client, cloud, codes)
	if err != nil {
		fmt.Fprintln(os.Stderr, "alarmcheck failed:", err)
		if len(records) > 0 {
			printRecords(records)
		}
		os.Exit(1)
	}
	printRecords(records)
	fmt.Printf("\nOK: %d alerts ingested, remediated, sent to mock cloud, and verified\n", len(records))
	if cfg.KeepServer {
		fmt.Printf("mock cloud web kept alive: %s\n", cloud.WebURL())
		fmt.Println("press Ctrl+C to stop")
		waitForInterrupt()
	}
}

func parseFlags() config {
	cfg := config{}
	flag.StringVar(&cfg.OncallURL, "oncall-url", "http://127.0.0.1:6872", "oncall backend base URL")
	flag.StringVar(&cfg.FaultCodes, "fault-codes", "010A,010B,010C,010D,010E", "comma separated fault codes")
	flag.DurationVar(&cfg.Timeout, "timeout", 30*time.Second, "overall and per-request timeout")
	flag.DurationVar(&cfg.Delay, "delay", 0, "delay between fault submissions")
	flag.StringVar(&cfg.Operator, "operator", "alarmcheck", "operator name for auto remediation")
	flag.IntVar(&cfg.TimeoutSeconds, "command-timeout", 5, "cloud command timeout seconds")
	flag.Int64Var(&cfg.Seed, "seed", time.Now().UnixNano(), "random seed")
	flag.StringVar(&cfg.CloudAddr, "cloud-addr", "127.0.0.1:19090", "mock cloud fixed listen address")
	flag.StringVar(&cfg.CloudPath, "cloud-path", "/api/cloud/commands", "mock cloud command POST path")
	flag.BoolVar(&cfg.ServeOnly, "serve-only", false, "only run the fixed-port mock cloud web server")
	flag.BoolVar(&cfg.KeepServer, "keep-server", false, "keep the mock cloud web server running after checks")
	flag.Parse()
	cfg.CloudPath = normalizePath(cfg.CloudPath)
	return cfg
}

func runCheck(ctx context.Context, cfg config, client *http.Client, cloud *mockCloud, codes []string) ([]checkRecord, error) {
	records := make([]checkRecord, 0, len(codes))
	for i, code := range codes {
		if i > 0 && cfg.Delay > 0 {
			if err := sleepContext(ctx, cfg.Delay); err != nil {
				return records, err
			}
		}
		seq := i + 1
		profile := profileForFault(code)
		ingestRes, err := ingestAlert(ctx, client, cfg, profile, seq)
		if err != nil {
			return records, fmt.Errorf("%s ingest: %w", code, err)
		}
		if ingestRes.AlertID == "" {
			return records, fmt.Errorf("%s ingest returned empty alertId", code)
		}

		remediateRes, err := autoRemediate(ctx, client, cfg, cloud.URL(), ingestRes.AlertID, seq)
		if err != nil {
			return records, fmt.Errorf("%s auto-remediate alert=%s: %w", code, ingestRes.AlertID, err)
		}
		detail, err := getAlert(ctx, client, cfg, ingestRes.AlertID)
		if err != nil {
			return records, fmt.Errorf("%s get alert=%s: %w", code, ingestRes.AlertID, err)
		}
		cloudCall, ok := cloud.findByAlertID(ingestRes.AlertID)
		if !ok {
			return records, fmt.Errorf("%s alert=%s did not reach mock cloud", code, ingestRes.AlertID)
		}
		if err := verifyRecord(profile, ingestRes.AlertID, remediateRes.Result, detail.Alert, cloudCall); err != nil {
			return records, err
		}
		records = append(records, checkRecord{
			FaultCode:     code,
			AlertID:       ingestRes.AlertID,
			DeviceID:      remediateRes.Result.DeviceID,
			SolutionCode:  remediateRes.Result.SolutionCode,
			Command:       remediateRes.Result.Command,
			CloudAccepted: remediateRes.Result.CommandResult.Accepted,
			FinalState:    detail.Alert.State,
			CloudTraceID:  remediateRes.Result.CommandResult.CloudTraceID,
		})
	}
	return records, nil
}

func ingestAlert(ctx context.Context, client *http.Client, cfg config, profile faultProfile, seq int) (ingestResponse, error) {
	payload := buildPayload(profile, seq, cfg.Seed)
	var response ingestResponse
	if err := postJSON(ctx, client, apiURL(cfg.OncallURL, "/alerts/ingest"), payload, &response); err != nil {
		return ingestResponse{}, err
	}
	return response, nil
}

func autoRemediate(ctx context.Context, client *http.Client, cfg config, cloudURL, alertID string, seq int) (remediateResponse, error) {
	payload := remediateRequest{
		CloudURL:       cloudURL,
		Operator:       cfg.Operator,
		TimeoutSeconds: cfg.TimeoutSeconds,
		Payload: map[string]interface{}{
			"alarmcheck": true,
			"sequence":   seq,
		},
	}
	var response remediateResponse
	if err := postJSON(ctx, client, apiURL(cfg.OncallURL, "/alerts/"+url.PathEscape(alertID)+"/auto-remediate"), payload, &response); err != nil {
		return remediateResponse{}, err
	}
	return response, nil
}

func getAlert(ctx context.Context, client *http.Client, cfg config, alertID string) (alertDetailResponse, error) {
	var response alertDetailResponse
	if err := getJSON(ctx, client, apiURL(cfg.OncallURL, "/alerts/"+url.PathEscape(alertID)), &response); err != nil {
		return alertDetailResponse{}, err
	}
	return response, nil
}

func postJSON(ctx context.Context, client *http.Client, endpoint string, payload interface{}, dataOut interface{}) error {
	body, err := json.Marshal(payload)
	if err != nil {
		return err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(body))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("User-Agent", "alarmcheck/1.0")
	return doJSON(client, req, dataOut)
}

func getJSON(ctx context.Context, client *http.Client, endpoint string, dataOut interface{}) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, endpoint, nil)
	if err != nil {
		return err
	}
	req.Header.Set("User-Agent", "alarmcheck/1.0")
	return doJSON(client, req, dataOut)
}

func doJSON(client *http.Client, req *http.Request, dataOut interface{}) error {
	resp, err := client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	body, _ := io.ReadAll(io.LimitReader(resp.Body, 2*1024*1024))
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return fmt.Errorf("%s %s returned %d: %s", req.Method, req.URL.String(), resp.StatusCode, strings.TrimSpace(string(body)))
	}

	var envelope apiEnvelope
	if err := json.Unmarshal(body, &envelope); err != nil {
		return fmt.Errorf("decode envelope: %w body=%s", err, trimForError(body))
	}
	if !strings.EqualFold(strings.TrimSpace(envelope.Message), "OK") {
		return fmt.Errorf("api message=%q body=%s", envelope.Message, trimForError(body))
	}
	if dataOut == nil {
		return nil
	}
	if len(bytes.TrimSpace(envelope.Data)) == 0 || bytes.Equal(bytes.TrimSpace(envelope.Data), []byte("null")) {
		return errors.New("api data is empty")
	}
	if err := json.Unmarshal(envelope.Data, dataOut); err != nil {
		return fmt.Errorf("decode data: %w data=%s", err, trimForError(envelope.Data))
	}
	return nil
}

func verifyRecord(profile faultProfile, alertID string, result remediateResult, detail alert, cloudCall cloudCommand) error {
	if !strings.EqualFold(result.AlertID, alertID) {
		return fmt.Errorf("%s result alertId mismatch: got %s want %s", profile.Code, result.AlertID, alertID)
	}
	if !strings.EqualFold(result.FaultCode, profile.Code) {
		return fmt.Errorf("%s result faultCode mismatch: got %s", profile.Code, result.FaultCode)
	}
	if result.SolutionCode != profile.SolutionCode {
		return fmt.Errorf("%s solutionCode mismatch: got %d want %d", profile.Code, result.SolutionCode, profile.SolutionCode)
	}
	if result.Command != profile.Command {
		return fmt.Errorf("%s command mismatch: got %s want %s", profile.Code, result.Command, profile.Command)
	}
	if !result.CommandResult.Accepted {
		return fmt.Errorf("%s cloud command was not accepted: %+v", profile.Code, result.CommandResult)
	}
	if detail.State != profile.ExpectedState {
		return fmt.Errorf("%s final state mismatch: got %s want %s", profile.Code, detail.State, profile.ExpectedState)
	}
	if cloudCall.Command != profile.Command {
		return fmt.Errorf("%s mock cloud command mismatch: got %s want %s", profile.Code, cloudCall.Command, profile.Command)
	}
	if cloudCall.FaultCode != profile.Code {
		return fmt.Errorf("%s mock cloud faultCode mismatch: got %s", profile.Code, cloudCall.FaultCode)
	}
	if solutionValue, ok := intFromAny(cloudCall.Payload["solutionCode"]); !ok || solutionValue != profile.SolutionCode {
		return fmt.Errorf("%s cloud payload solutionCode mismatch: got %v want %d", profile.Code, cloudCall.Payload["solutionCode"], profile.SolutionCode)
	}
	return nil
}

func buildPayload(profile faultProfile, seq int, seed int64) ingestPayload {
	rng := rand.New(rand.NewSource(seed + int64(seq)))
	now := time.Now().UTC()
	region := []string{"华东一区", "华南一区", "华北一区", "西南一区"}[rng.Intn(4)]
	siteID := []string{"site-sh-01", "site-sz-02", "site-bj-03", "site-hz-04"}[rng.Intn(4)]
	city := []string{"SH", "SZ", "HZ", "BJ", "CD"}[rng.Intn(5)]
	deviceID := fmt.Sprintf("HS-INV-%s-CHECK-%03d", city, seq)
	gatewayID := fmt.Sprintf("GW-%s-CHECK-%03d", city, rng.Intn(30)+1)
	eventID := fmt.Sprintf("alarmcheck-%s-%03d-%s", now.Format("20060102-150405"), seq, strings.ToLower(profile.Code))

	return ingestPayload{
		Source:        "device-cloud",
		SourceEventID: eventID,
		SourceAlertID: fmt.Sprintf("%s:%s:%d", deviceID, profile.Code, now.UnixNano()),
		EventType:     "firing",
		Actor:         "alarmcheck",
		OccurredAt:    now.Add(-time.Duration(rng.Intn(180)) * time.Second),
		Alert: externalAlertPayload{
			Name:          profile.Name,
			FaultCode:     profile.Code,
			FaultName:     profile.Name,
			FaultCategory: profile.Category,
			Severity:      profile.Severity,
			Service:       "energy-storage",
			Team:          "ops",
			Region:        region,
			Environment:   "prod",
			DeviceID:      deviceID,
			ProductKey:    "HS-ESS25K",
			GatewayID:     gatewayID,
			MetricName:    profile.MetricName,
			MetricValue:   profile.MetricValue,
			Threshold:     profile.Threshold,
			Unit:          profile.Unit,
			Description:   fmt.Sprintf("%s %s 当前值 %s%s，阈值 %s%s", deviceID, profile.Name, profile.MetricValue, profile.Unit, profile.Threshold, profile.Unit),
			StartsAt:      now.Add(-time.Duration(3+rng.Intn(10)) * time.Minute),
			Labels: map[string]string{
				"site_id":     siteID,
				"device_type": "海索逆变器",
				"cabinet_id":  fmt.Sprintf("cabinet-%02d", rng.Intn(12)+1),
				"firmware":    fmt.Sprintf("v1.%d.%d", rng.Intn(9)+1, rng.Intn(10)),
			},
			Annotations: map[string]string{
				"fault_name_en": profile.EnglishName,
				"alarm_level":   profile.AlarmLevel,
			},
			Attributes: map[string]interface{}{
				"mock":            true,
				"alarmcheck":      true,
				"sequence":        seq,
				"faultCatalog":    "海索逆变器",
				"faultNameEn":     profile.EnglishName,
				"faultLevelValue": profile.FaultLevel,
			},
		},
		Raw: map[string]interface{}{
			"mock":       true,
			"alarmcheck": true,
			"sequence":   seq,
		},
	}
}

func newMockCloud(addr, commandPath string) (*mockCloud, error) {
	commandPath = normalizePath(commandPath)
	listener, err := net.Listen("tcp", addr)
	if err != nil {
		return nil, err
	}
	cloud := &mockCloud{
		addr:        listener.Addr().String(),
		commandPath: commandPath,
		baseURL:     "http://" + listenHost(listener.Addr()),
		listener:    listener,
	}
	mux := http.NewServeMux()
	mux.HandleFunc("/", cloud.handleDashboard)
	mux.HandleFunc("/api/commands", cloud.handleCommandsAPI)
	mux.HandleFunc("/healthz", cloud.handleHealth)
	mux.HandleFunc(commandPath, cloud.handleCommandPost)
	cloud.server = &http.Server{
		Handler:           mux,
		ReadHeaderTimeout: 5 * time.Second,
	}
	go func() {
		if err := cloud.server.Serve(listener); err != nil && !errors.Is(err, http.ErrServerClosed) {
			fmt.Fprintln(os.Stderr, "mock cloud server stopped:", err)
		}
	}()
	return cloud, nil
}

func (m *mockCloud) URL() string {
	return m.baseURL + m.commandPath
}

func (m *mockCloud) WebURL() string {
	return m.baseURL + "/"
}

func (m *mockCloud) Close() {
	if m == nil || m.server == nil {
		return
	}
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	_ = m.server.Shutdown(ctx)
}

func (m *mockCloud) findByAlertID(alertID string) (cloudCommand, bool) {
	m.mu.Lock()
	defer m.mu.Unlock()
	for _, record := range m.records {
		if record.Command.AlertID == alertID {
			return record.Command, true
		}
	}
	return cloudCommand{}, false
}

func (m *mockCloud) handleCommandPost(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	defer r.Body.Close()
	body, _ := io.ReadAll(io.LimitReader(r.Body, 2*1024*1024))
	var command cloudCommand
	if err := json.Unmarshal(body, &command); err != nil {
		http.Error(w, "bad json", http.StatusBadRequest)
		return
	}
	record := cloudCommandRecord{
		ReceivedAt: time.Now().UTC(),
		RemoteAddr: r.RemoteAddr,
		UserAgent:  r.UserAgent(),
		Command:    command,
		Raw:        append(json.RawMessage(nil), body...),
	}
	m.mu.Lock()
	record.Index = len(m.records) + 1
	m.records = append(m.records, record)
	count := len(m.records)
	m.mu.Unlock()

	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]interface{}{
		"code":     "0",
		"success":  true,
		"accepted": true,
		"status":   "accepted",
		"message":  "mock cloud accepted command",
		"traceId":  fmt.Sprintf("mock-cloud-%03d", count),
	})
}

func (m *mockCloud) handleCommandsAPI(w http.ResponseWriter, r *http.Request) {
	switch r.Method {
	case http.MethodGet:
		m.mu.Lock()
		records := append([]cloudCommandRecord(nil), m.records...)
		m.mu.Unlock()
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]interface{}{
			"count":   len(records),
			"records": records,
		})
	case http.MethodDelete:
		m.mu.Lock()
		m.records = nil
		m.mu.Unlock()
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]interface{}{
			"cleared": true,
		})
	default:
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
	}
}

func (m *mockCloud) handleHealth(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]interface{}{
		"status":          "ok",
		"webUrl":          m.WebURL(),
		"commandEndpoint": m.URL(),
	})
}

func (m *mockCloud) handleDashboard(w http.ResponseWriter, r *http.Request) {
	if r.URL.Path != "/" {
		http.NotFound(w, r)
		return
	}
	m.mu.Lock()
	records := append([]cloudCommandRecord(nil), m.records...)
	m.mu.Unlock()
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	_ = mockCloudTemplate.Execute(w, map[string]interface{}{
		"CommandEndpoint": m.URL(),
		"Records":         records,
		"Count":           len(records),
	})
}

func profileForFault(code string) faultProfile {
	code = strings.ToUpper(strings.TrimSpace(code))
	if profile, ok := knownFaultProfiles()[code]; ok {
		return profile
	}
	solutionCode := solutionCodeForFault(code)
	command, finalState := expectedCommandForSolution(solutionCode)
	return faultProfile{
		Code:          code,
		Name:          "海索逆变器故障告警",
		EnglishName:   "Haiso Inverter Fault Alert",
		Category:      "inverter",
		MetricName:    "fault_signal",
		MetricValue:   "1",
		Threshold:     "1",
		Unit:          "count",
		Severity:      "warning",
		AlarmLevel:    "1级-高级",
		FaultLevel:    2,
		SolutionCode:  solutionCode,
		Command:       command,
		ExpectedState: finalState,
	}
}

func solutionCodeForFault(code string) int {
	code = strings.ToUpper(strings.TrimSpace(code))
	for i := len(code) - 1; i >= 0; i-- {
		value, err := strconv.ParseInt(string(code[i]), 16, 64)
		if err == nil {
			return int(value % 6)
		}
	}
	return 0
}

func expectedCommandForSolution(code int) (string, string) {
	switch code {
	case 0:
		return "restart_inverter", "mitigated"
	case 1:
		return "restart_iot_gateway", "mitigated"
	case 2:
		return "dispatch_field_ops", "mitigated"
	case 3:
		return "observe_only", "investigating"
	case 4:
		return "reset_after_battery_check", "mitigated"
	case 5:
		return "escalate_l2_ops", "mitigated"
	default:
		return "", "mitigated"
	}
}

func knownFaultProfiles() map[string]faultProfile {
	return map[string]faultProfile{
		"010A": {
			Code:          "010A",
			Name:          "dcdc1 温度采样错误",
			EnglishName:   "DCDC1 Temperature Sensing Error",
			Category:      "thermal",
			MetricName:    "temperature",
			MetricValue:   "91.8",
			Threshold:     "80",
			Unit:          "C",
			Severity:      "critical",
			AlarmLevel:    "1级-高级",
			FaultLevel:    2,
			SolutionCode:  4,
			Command:       "reset_after_battery_check",
			ExpectedState: "mitigated",
		},
		"010B": {
			Code:          "010B",
			Name:          "dcdc2 温度采样错误",
			EnglishName:   "DCDC2 Temperature Sensing Error",
			Category:      "thermal",
			MetricName:    "temperature",
			MetricValue:   "92.1",
			Threshold:     "80",
			Unit:          "C",
			Severity:      "critical",
			AlarmLevel:    "1级-高级",
			FaultLevel:    2,
			SolutionCode:  5,
			Command:       "escalate_l2_ops",
			ExpectedState: "mitigated",
		},
		"010C": {
			Code:          "010C",
			Name:          "dcdc3 温度采样错误",
			EnglishName:   "DCDC3 Temperature Sensing Error",
			Category:      "thermal",
			MetricName:    "temperature",
			MetricValue:   "90.5",
			Threshold:     "80",
			Unit:          "C",
			Severity:      "critical",
			AlarmLevel:    "1级-高级",
			FaultLevel:    2,
			SolutionCode:  0,
			Command:       "restart_inverter",
			ExpectedState: "mitigated",
		},
		"010D": {
			Code:          "010D",
			Name:          "1.05 倍过载",
			EnglishName:   "105% Overloaded",
			Category:      "overload",
			MetricName:    "load_ratio",
			MetricValue:   "108",
			Threshold:     "105",
			Unit:          "%",
			Severity:      "critical",
			AlarmLevel:    "1级-高级",
			FaultLevel:    3,
			SolutionCode:  1,
			Command:       "restart_iot_gateway",
			ExpectedState: "mitigated",
		},
		"010E": {
			Code:          "010E",
			Name:          "1.2 倍过载",
			EnglishName:   "120% Overloaded",
			Category:      "overload",
			MetricName:    "load_ratio",
			MetricValue:   "126",
			Threshold:     "120",
			Unit:          "%",
			Severity:      "critical",
			AlarmLevel:    "1级-高级",
			FaultLevel:    3,
			SolutionCode:  2,
			Command:       "dispatch_field_ops",
			ExpectedState: "mitigated",
		},
		"010F": haisoFaultProfile("010F", "2 倍过载", "200% Overloaded", "overload", "load_ratio", "205", "200", "%", "critical", "1级-高级", 3, 3),
		"011A": haisoFaultProfile("011A", "逆变散热器过温", "Inverter Radiator Overtemperature", "thermal", "temperature", "88", "80", "C", "critical", "1级-高级", 2, 4),
		"011B": haisoFaultProfile("011B", "逆变散热器故障", "Inverter Radiator Failure", "cooling", "radiator_status", "1", "1", "count", "critical", "1级-高级", 2, 5),
		"012A": haisoFaultProfile("012A", "电网包络异常", "Abnormal Grid Envelope", "grid", "grid_voltage", "262", "260", "V", "warning", "1级-低级", 2, 4),
		"012B": haisoFaultProfile("012B", "电网锁相异常", "Abnormal Phase Locking of Grid", "grid", "grid_frequency", "49.2", "50", "Hz", "critical", "1级-高级", 2, 5),
		"012C": haisoFaultProfile("012C", "缓冲继电器粘死检测", "Abnormal Detection of Stuck Buffer Relay", "relay", "relay_status", "1", "1", "count", "critical", "1级-高级", 2, 0),
		"012D": haisoFaultProfile("012D", "主继电器粘死检测", "Abnormal Detection of Main Relay Sticking", "relay", "relay_status", "1", "1", "count", "critical", "1级-高级", 2, 1),
		"013A": haisoFaultProfile("013A", "变压器故障", "Transformer Fault", "transformer", "transformer_status", "1", "1", "count", "critical", "1级-高级", 2, 4),
		"013B": haisoFaultProfile("013B", "通讯故障", "Inverter Communication Failure", "communication", "heartbeat_loss", "4", "3", "count", "critical", "1级-高级", 2, 5),
		"013C": haisoFaultProfile("013C", "风机故障", "Inverter Fans Fault", "cooling", "fan_speed", "500", "800", "rpm", "critical", "1级-高级", 2, 0),
		"015A": haisoFaultProfile("015A", "dcdc1 温度采样错误", "DCDC1 Temperature Sensing Error", "thermal", "temperature", "91.8", "80", "C", "critical", "1级-高级", 2, 4),
		"015B": haisoFaultProfile("015B", "dcdc2 温度采样错误", "DCDC2 Temperature Sensing Error", "thermal", "temperature", "92.1", "80", "C", "critical", "1级-高级", 2, 5),
		"015C": haisoFaultProfile("015C", "dcdc3 温度采样错误", "DCDC3 Temperature Sensing Error", "thermal", "temperature", "90.5", "80", "C", "critical", "1级-高级", 2, 0),
		"015D": haisoFaultProfile("015D", "1.05 倍过载", "105% Overloaded", "overload", "load_ratio", "108", "105", "%", "critical", "1级-高级", 3, 1),
		"015E": haisoFaultProfile("015E", "1.2 倍过载", "120% Overloaded", "overload", "load_ratio", "126", "120", "%", "critical", "1级-高级", 3, 2),
	}
}

func haisoFaultProfile(code, name, englishName, category, metricName, metricValue, threshold, unit, severity, alarmLevel string, faultLevel, solutionCode int) faultProfile {
	command, finalState := expectedCommandForSolution(solutionCode)
	return faultProfile{
		Code:          code,
		Name:          name,
		EnglishName:   englishName,
		Category:      category,
		MetricName:    metricName,
		MetricValue:   metricValue,
		Threshold:     threshold,
		Unit:          unit,
		Severity:      severity,
		AlarmLevel:    alarmLevel,
		FaultLevel:    faultLevel,
		SolutionCode:  solutionCode,
		Command:       command,
		ExpectedState: finalState,
	}
}

func apiURL(base, apiPath string) string {
	base = strings.TrimRight(strings.TrimSpace(base), "/")
	if base == "" {
		base = "http://127.0.0.1:6872"
	}
	if !strings.HasPrefix(apiPath, "/") {
		apiPath = "/" + apiPath
	}
	if strings.HasSuffix(base, "/api") {
		return base + apiPath
	}
	return base + "/api" + apiPath
}

func parseFaultCodes(spec string) []string {
	parts := strings.FieldsFunc(spec, func(r rune) bool {
		return r == ',' || r == ';' || r == ' ' || r == '\n' || r == '\t'
	})
	codes := make([]string, 0, len(parts))
	seen := map[string]bool{}
	for _, part := range parts {
		code := strings.ToUpper(strings.TrimSpace(part))
		if code == "" || seen[code] {
			continue
		}
		seen[code] = true
		codes = append(codes, code)
	}
	return codes
}

func normalizePath(path string) string {
	path = strings.TrimSpace(path)
	if path == "" {
		return "/api/cloud/commands"
	}
	if !strings.HasPrefix(path, "/") {
		path = "/" + path
	}
	return path
}

func listenHost(addr net.Addr) string {
	tcpAddr, ok := addr.(*net.TCPAddr)
	if !ok {
		return addr.String()
	}
	host := tcpAddr.IP.String()
	if tcpAddr.IP == nil || tcpAddr.IP.IsUnspecified() {
		host = "127.0.0.1"
	}
	return net.JoinHostPort(host, strconv.Itoa(tcpAddr.Port))
}

func waitForInterrupt() {
	ch := make(chan os.Signal, 1)
	signal.Notify(ch, os.Interrupt, syscall.SIGTERM)
	<-ch
}

func intFromAny(value interface{}) (int, bool) {
	switch v := value.(type) {
	case int:
		return v, true
	case float64:
		return int(v), true
	case json.Number:
		i, err := strconv.Atoi(v.String())
		return i, err == nil
	default:
		return 0, false
	}
}

func sleepContext(ctx context.Context, delay time.Duration) error {
	timer := time.NewTimer(delay)
	defer timer.Stop()
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-timer.C:
		return nil
	}
}

func trimForError(body []byte) string {
	text := strings.TrimSpace(string(body))
	if len(text) > 800 {
		return text[:800] + "..."
	}
	return text
}

func prettyJSON(value interface{}) string {
	switch v := value.(type) {
	case json.RawMessage:
		var decoded interface{}
		if err := json.Unmarshal(v, &decoded); err == nil {
			value = decoded
		}
	case []byte:
		var decoded interface{}
		if err := json.Unmarshal(v, &decoded); err == nil {
			value = decoded
		}
	}
	body, err := json.MarshalIndent(value, "", "  ")
	if err != nil {
		return fmt.Sprint(value)
	}
	return string(body)
}

func printRecords(records []checkRecord) {
	sort.SliceStable(records, func(i, j int) bool {
		return records[i].FaultCode < records[j].FaultCode
	})
	fmt.Println("faultCode\talertId\tdeviceId\tsolutionCode\tcommand\tcloudAccepted\tfinalState\tcloudTraceId")
	for _, record := range records {
		fmt.Printf(
			"%s\t%s\t%s\t%d\t%s\t%t\t%s\t%s\n",
			record.FaultCode,
			record.AlertID,
			record.DeviceID,
			record.SolutionCode,
			record.Command,
			record.CloudAccepted,
			record.FinalState,
			record.CloudTraceID,
		)
	}
}
