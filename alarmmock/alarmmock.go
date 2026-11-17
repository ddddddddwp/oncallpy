package main

import (
	"bytes"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"math"
	"math/rand"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"
)

type config struct {
	URL           string
	Count         int
	Concurrency   int
	FaultCode     string
	FaultName     string
	FaultCategory string
	Severity      string
	Source        string
	Mode          string
	Timeout       time.Duration
	QPS           int
	Seed          int64
	SameAlert     bool
	PrintSample   bool
}

type ingestPayload struct {
	Source        string                 `json:"source"`
	SourceEventID string                 `json:"sourceEventId"`
	SourceAlertID string                 `json:"sourceAlertId"`
	EventType     string                 `json:"eventType"`
	OccurredAt    time.Time              `json:"occurredAt"`
	Alert         externalAlertPayload   `json:"alert"`
	Raw           map[string]interface{} `json:"raw,omitempty"`
}

type externalAlertPayload struct {
	Name          string            `json:"name"`
	FaultCode     string            `json:"faultCode"`
	FaultName     string            `json:"faultName"`
	FaultCategory string            `json:"faultCategory"`
	Severity      string            `json:"severity"`
	Service       string            `json:"service"`
	Team          string            `json:"team"`
	Region        string            `json:"region"`
	Environment   string            `json:"environment"`
	DeviceID      string            `json:"deviceId"`
	ProductKey    string            `json:"productKey"`
	GatewayID     string            `json:"gatewayId"`
	MetricName    string            `json:"metricName"`
	MetricValue   string            `json:"metricValue"`
	Threshold     string            `json:"threshold"`
	Unit          string            `json:"unit"`
	Description   string            `json:"description"`
	Labels        map[string]string `json:"labels,omitempty"`
	Annotations   map[string]string `json:"annotations,omitempty"`
	Attributes    map[string]any    `json:"attributes,omitempty"`
}

type faultProfile struct {
	Code        string
	Name        string
	EnglishName string
	Category    string
	MetricName  string
	Threshold   string
	Unit        string
	DeviceType  string
	AlarmLevel  string
	LevelValue  int
}

type result struct {
	Sequence   int
	StatusCode int
	Action     string
	Latency    time.Duration
	Error      string
	Body       string
}

type stats struct {
	mu         sync.Mutex
	total      int
	success    int
	failed     int
	status     map[int]int
	actions    map[string]int
	errors     map[string]int
	latencies  []time.Duration
	failSample []result
}

func main() {
	cfg := parseFlags()
	if cfg.FaultCode == "" {
		fmt.Fprintln(os.Stderr, "-fault-code is required")
		os.Exit(2)
	}
	if cfg.Count <= 0 || cfg.Concurrency <= 0 {
		fmt.Fprintln(os.Stderr, "-count and -concurrency must be positive")
		os.Exit(2)
	}

	client := &http.Client{
		Timeout: cfg.Timeout,
		Transport: &http.Transport{
			MaxIdleConns:        cfg.Concurrency * 2,
			MaxIdleConnsPerHost: cfg.Concurrency * 2,
			IdleConnTimeout:     30 * time.Second,
		},
	}
	if cfg.PrintSample {
		printSamplePayloads(cfg)
	}

	start := time.Now()
	summary := run(context.Background(), cfg, client)
	printStats(summary, time.Since(start))
	if summary.failed > 0 {
		os.Exit(1)
	}
}

func parseFlags() config {
	cfg := config{}
	flag.StringVar(&cfg.URL, "url", "http://127.0.0.1:6872/api/alerts/ingest", "oncall alert ingest URL")
	flag.IntVar(&cfg.Count, "count", 100, "total alert count")
	flag.IntVar(&cfg.Concurrency, "concurrency", 10, "concurrent workers")
	flag.StringVar(&cfg.FaultCode, "fault-code", "", "required fault code, for example 010A, E_TEMP_001, random, or 010A,0116,1159")
	flag.StringVar(&cfg.FaultName, "fault-name", "", "fault display name")
	flag.StringVar(&cfg.FaultCategory, "fault-category", "", "fault category")
	flag.StringVar(&cfg.Severity, "severity", "warning", "alert severity")
	flag.StringVar(&cfg.Source, "source", "device-cloud", "alert source")
	flag.StringVar(&cfg.Mode, "mode", "firing", "event mode: firing, resolved, mixed")
	flag.DurationVar(&cfg.Timeout, "timeout", 5*time.Second, "per-request timeout")
	flag.IntVar(&cfg.QPS, "qps", 0, "requests per second limit, 0 means unlimited")
	flag.Int64Var(&cfg.Seed, "seed", time.Now().UnixNano(), "random seed")
	flag.BoolVar(&cfg.SameAlert, "same-alert", false, "send the same device and fault repeatedly")
	flag.BoolVar(&cfg.PrintSample, "print-sample", false, "print first sample payloads")
	flag.Parse()
	cfg.Mode = strings.ToLower(strings.TrimSpace(cfg.Mode))
	if cfg.Mode == "" {
		cfg.Mode = "firing"
	}
	return cfg
}

func run(ctx context.Context, cfg config, client *http.Client) *stats {
	jobs := make(chan int)
	results := make(chan result)
	summary := &stats{
		status:  map[int]int{},
		actions: map[string]int{},
		errors:  map[string]int{},
	}

	var ticker *time.Ticker
	var throttle <-chan time.Time
	if cfg.QPS > 0 {
		interval := time.Second / time.Duration(cfg.QPS)
		if interval <= 0 {
			interval = time.Nanosecond
		}
		ticker = time.NewTicker(interval)
		throttle = ticker.C
		defer ticker.Stop()
	}

	var wg sync.WaitGroup
	for i := 0; i < cfg.Concurrency; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for seq := range jobs {
				if throttle != nil {
					select {
					case <-ctx.Done():
						results <- result{Sequence: seq, Error: ctx.Err().Error()}
						continue
					case <-throttle:
					}
				}
				results <- sendOne(ctx, client, cfg, seq)
			}
		}()
	}

	go func() {
		for i := 1; i <= cfg.Count; i++ {
			jobs <- i
		}
		close(jobs)
		wg.Wait()
		close(results)
	}()

	for res := range results {
		summary.add(res)
	}
	return summary
}

func sendOne(ctx context.Context, client *http.Client, cfg config, seq int) result {
	payload := buildPayload(cfg, seq)
	body, err := json.Marshal(payload)
	if err != nil {
		return result{Sequence: seq, Error: err.Error()}
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, cfg.URL, bytes.NewReader(body))
	if err != nil {
		return result{Sequence: seq, Error: err.Error()}
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("User-Agent", "alarmmock/1.0")

	start := time.Now()
	resp, err := client.Do(req)
	latency := time.Since(start)
	if err != nil {
		return result{Sequence: seq, Latency: latency, Error: err.Error()}
	}
	defer resp.Body.Close()

	respBody, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
	action := extractAction(respBody)
	res := result{
		Sequence:   seq,
		StatusCode: resp.StatusCode,
		Action:     action,
		Latency:    latency,
		Body:       string(respBody),
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		res.Error = fmt.Sprintf("non-2xx status %d", resp.StatusCode)
	}
	return res
}

func buildPayload(cfg config, seq int) ingestPayload {
	rng := rand.New(rand.NewSource(cfg.Seed + int64(seq)))
	faultCode := selectFaultCode(cfg.FaultCode, rng)
	profile := profileForFault(faultCode)
	if cfg.FaultName != "" {
		profile.Name = cfg.FaultName
	}
	if cfg.FaultCategory != "" {
		profile.Category = cfg.FaultCategory
	}
	device := randomDevice(rng, cfg.SameAlert)
	if profile.DeviceType == "海索逆变器" {
		device = randomHaisoInverterDevice(rng, cfg.SameAlert)
	}
	region := []string{"华东一区", "华南一区", "华北一区", "西南一区"}[rng.Intn(4)]
	siteID := []string{"site-sh-01", "site-sz-02", "site-bj-03", "site-hz-04"}[rng.Intn(4)]
	value := metricValue(profile, rng)
	eventType := eventTypeForMode(cfg.Mode, rng)
	now := time.Now().UTC()

	return ingestPayload{
		Source:        cfg.Source,
		SourceEventID: fmt.Sprintf("mock-%s-%06d", now.Format("20060102-150405"), seq),
		SourceAlertID: fmt.Sprintf("%s:%s", device.id, faultCode),
		EventType:     eventType,
		OccurredAt:    now.Add(-time.Duration(rng.Intn(300)) * time.Second),
		Alert: externalAlertPayload{
			Name:          profile.Name,
			FaultCode:     faultCode,
			FaultName:     profile.Name,
			FaultCategory: profile.Category,
			Severity:      cfg.Severity,
			Service:       "energy-storage",
			Team:          "ops",
			Region:        region,
			Environment:   "prod",
			DeviceID:      device.id,
			ProductKey:    device.productKey,
			GatewayID:     device.gatewayID,
			MetricName:    profile.MetricName,
			MetricValue:   value,
			Threshold:     profile.Threshold,
			Unit:          profile.Unit,
			Description:   fmt.Sprintf("%s %s %s%s，超过阈值 %s%s", device.id, profile.Name, value, profile.Unit, profile.Threshold, profile.Unit),
			Labels: map[string]string{
				"site_id":     siteID,
				"device_type": firstNonEmptyString(profile.DeviceType, device.kind),
				"cabinet_id":  fmt.Sprintf("cabinet-%02d", rng.Intn(12)+1),
				"firmware":    fmt.Sprintf("v1.%d.%d", rng.Intn(9)+1, rng.Intn(10)),
			},
			Annotations: map[string]string{
				"fault_name_en": profile.EnglishName,
				"alarm_level":   profile.AlarmLevel,
			},
			Attributes: map[string]any{
				"mock":            true,
				"sequence":        seq,
				"faultCatalog":    profile.DeviceType,
				"faultNameEn":     profile.EnglishName,
				"alarmLevel":      profile.AlarmLevel,
				"faultLevelValue": profile.LevelValue,
			},
		},
		Raw: map[string]interface{}{
			"mock":     true,
			"sequence": seq,
		},
	}
}

type deviceInfo struct {
	id         string
	kind       string
	productKey string
	gatewayID  string
}

func randomDevice(rng *rand.Rand, same bool) deviceInfo {
	if same {
		return deviceInfo{id: "ESS-SH-001", kind: "inverter", productKey: "HS-ESS25K", gatewayID: "GW-SH-001"}
	}
	prefixes := []deviceInfo{
		{id: "ESS", kind: "inverter", productKey: "HS-ESS25K", gatewayID: "GW-SH"},
		{id: "BMS", kind: "battery", productKey: "BMS-STACK", gatewayID: "GW-HZ"},
		{id: "PCS", kind: "converter", productKey: "PCS-100K", gatewayID: "GW-BJ"},
		{id: "INV", kind: "inverter", productKey: "INV-50K", gatewayID: "GW-SZ"},
	}
	base := prefixes[rng.Intn(len(prefixes))]
	city := []string{"SH", "SZ", "HZ", "BJ"}[rng.Intn(4)]
	num := rng.Intn(300) + 1
	return deviceInfo{
		id:         fmt.Sprintf("%s-%s-%03d", base.id, city, num),
		kind:       base.kind,
		productKey: base.productKey,
		gatewayID:  fmt.Sprintf("%s-%03d", base.gatewayID, rng.Intn(40)+1),
	}
}

func randomHaisoInverterDevice(rng *rand.Rand, same bool) deviceInfo {
	if same {
		return deviceInfo{id: "HS-INV-SH-001", kind: "haiso-inverter", productKey: "HS-ESS25K", gatewayID: "GW-SH-001"}
	}
	city := []string{"SH", "SZ", "HZ", "BJ", "CD"}[rng.Intn(5)]
	return deviceInfo{
		id:         fmt.Sprintf("HS-INV-%s-%03d", city, rng.Intn(300)+1),
		kind:       "haiso-inverter",
		productKey: "HS-ESS25K",
		gatewayID:  fmt.Sprintf("GW-%s-%03d", city, rng.Intn(40)+1),
	}
}

func selectFaultCode(spec string, rng *rand.Rand) string {
	spec = strings.TrimSpace(spec)
	lower := strings.ToLower(spec)
	if lower == "random" || lower == "haiso" || lower == "haiso-random" {
		codes := haisoFaultCodes()
		if len(codes) == 0 {
			return "010A"
		}
		return codes[rng.Intn(len(codes))]
	}
	codes := splitFaultCodeSpec(spec)
	if len(codes) > 1 {
		return codes[rng.Intn(len(codes))]
	}
	if len(codes) == 1 {
		return codes[0]
	}
	return strings.ToUpper(spec)
}

func splitFaultCodeSpec(spec string) []string {
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

func profileForFault(code string) faultProfile {
	code = strings.ToUpper(strings.TrimSpace(code))
	if profile, ok := loadHaisoFaultCatalog()[code]; ok {
		return profile
	}
	profiles := map[string]faultProfile{
		"E_TEMP_001":  fallbackProfile(code, "设备温度过高", "Device Temperature High", "thermal", "temperature", "80", "C"),
		"E_BATT_LOW":  fallbackProfile(code, "电池电量低", "Battery SOC Low", "battery", "battery_soc", "20", "%"),
		"E_VOLT_HIGH": fallbackProfile(code, "电压过高", "Voltage High", "electrical", "voltage", "750", "V"),
		"E_NET_LOST":  fallbackProfile(code, "设备离线", "Device Offline", "network", "heartbeat_loss", "3", "count"),
		"E_FAN_FAULT": fallbackProfile(code, "风扇异常", "Fans Abnormal", "cooling", "fan_speed", "800", "rpm"),
	}
	if profile, ok := profiles[code]; ok {
		return profile
	}
	return fallbackProfile(code, "设备故障告警", "Device Fault Alert", "generic", "fault_signal", "1", "count")
}

var (
	haisoCatalogOnce sync.Once
	haisoCatalog     map[string]faultProfile
	haisoCodes       []string
)

func loadHaisoFaultCatalog() map[string]faultProfile {
	haisoCatalogOnce.Do(func() {
		haisoCatalog = parseHaisoFaultDoc()
		haisoCodes = make([]string, 0, len(haisoCatalog))
		for code := range haisoCatalog {
			haisoCodes = append(haisoCodes, code)
		}
		sort.Strings(haisoCodes)
	})
	return haisoCatalog
}

func haisoFaultCodes() []string {
	_ = loadHaisoFaultCatalog()
	return append([]string(nil), haisoCodes...)
}

func parseHaisoFaultDoc() map[string]faultProfile {
	path := findHaisoFaultDoc()
	if path == "" {
		return fallbackHaisoFaultCatalog()
	}
	content, err := os.ReadFile(path)
	if err != nil {
		return fallbackHaisoFaultCatalog()
	}
	catalog := map[string]faultProfile{}
	for _, line := range strings.Split(string(content), "\n") {
		if !strings.HasPrefix(line, "| ") || !strings.Contains(line, "海索逆变器") {
			continue
		}
		cols := strings.Split(line, "|")
		if len(cols) < 10 {
			continue
		}
		code := strings.ToUpper(strings.TrimSpace(cols[4]))
		if code == "" || code == "故障码" {
			continue
		}
		name := strings.TrimSpace(cols[5])
		englishName := strings.TrimSpace(cols[6])
		levelValue, _ := strconv.Atoi(strings.TrimSpace(cols[7]))
		category := categoryForFault(name, englishName)
		metricName, threshold, unit := metricForCategory(category, name)
		catalog[code] = faultProfile{
			Code:        code,
			Name:        name,
			EnglishName: englishName,
			Category:    category,
			MetricName:  metricName,
			Threshold:   threshold,
			Unit:        unit,
			DeviceType:  strings.TrimSpace(cols[2]),
			AlarmLevel:  strings.TrimSpace(cols[3]),
			LevelValue:  levelValue,
		}
	}
	if len(catalog) == 0 {
		return fallbackHaisoFaultCatalog()
	}
	return catalog
}

func findHaisoFaultDoc() string {
	candidates := []string{
		"docs/海索逆变器故障码对照表.md",
		"OnCallAgent/docs/海索逆变器故障码对照表.md",
	}
	wd, err := os.Getwd()
	if err != nil {
		return ""
	}
	for {
		for _, candidate := range candidates {
			path := filepath.Join(wd, candidate)
			if _, err := os.Stat(path); err == nil {
				return path
			}
		}
		next := filepath.Dir(wd)
		if next == wd {
			return ""
		}
		wd = next
	}
}

func fallbackHaisoFaultCatalog() map[string]faultProfile {
	return map[string]faultProfile{
		"010A": haisoProfile("010A", "dcdc1 温度采样错误", "DCDC1 Temperature Sensing Error", "1级-高级", 2),
		"0116": haisoProfile("0116", "逆变输出短路", "Inverter Output Short Circuit", "1级-低级", 2),
		"013B": haisoProfile("013B", "通讯故障", "Inverter Communication Failure", "1级-高级", 2),
		"013C": haisoProfile("013C", "风机故障", "Inverter Fans Fault", "1级-高级", 2),
		"1102": haisoProfile("1102", "电池过温", "Battery Overtemperature", "1级-高级", 2),
		"1132": haisoProfile("1132", "系统全局关机", "Global System Shutdown", "1级-高级", 2),
		"1159": haisoProfile("1159", "光伏过压", "PV Overvoltage", "1级-高级", 2),
	}
}

func haisoProfile(code, name, englishName, alarmLevel string, levelValue int) faultProfile {
	category := categoryForFault(name, englishName)
	metricName, threshold, unit := metricForCategory(category, name)
	return faultProfile{
		Code:        code,
		Name:        name,
		EnglishName: englishName,
		Category:    category,
		MetricName:  metricName,
		Threshold:   threshold,
		Unit:        unit,
		DeviceType:  "海索逆变器",
		AlarmLevel:  alarmLevel,
		LevelValue:  levelValue,
	}
}

func fallbackProfile(code, name, englishName, category, metricName, threshold, unit string) faultProfile {
	return faultProfile{
		Code:        code,
		Name:        name,
		EnglishName: englishName,
		Category:    category,
		MetricName:  metricName,
		Threshold:   threshold,
		Unit:        unit,
		DeviceType:  "",
		AlarmLevel:  "",
		LevelValue:  0,
	}
}

func categoryForFault(name, englishName string) string {
	text := strings.ToLower(name + " " + englishName)
	switch {
	case strings.Contains(text, "通讯") || strings.Contains(text, "communication") || strings.Contains(text, "offline"):
		return "communication"
	case strings.Contains(text, "风机") || strings.Contains(text, "风扇") || strings.Contains(text, "fan"):
		return "cooling"
	case strings.Contains(text, "温") || strings.Contains(text, "temperature") || strings.Contains(text, "overtemperature"):
		return "thermal"
	case strings.Contains(text, "电池") || strings.Contains(text, "battery"):
		return "battery"
	case strings.Contains(text, "电网") || strings.Contains(text, "频率") || strings.Contains(text, "grid") || strings.Contains(text, "frequency"):
		return "grid"
	case strings.Contains(text, "光伏") || strings.Contains(text, "pv"):
		return "pv"
	case strings.Contains(text, "变压器") || strings.Contains(text, "transformer"):
		return "transformer"
	case strings.Contains(text, "绝缘") || strings.Contains(text, "漏电") || strings.Contains(text, "insulation") || strings.Contains(text, "leakage"):
		return "safety"
	case strings.Contains(text, "继电器") || strings.Contains(text, "relay"):
		return "relay"
	case strings.Contains(text, "过载") || strings.Contains(text, "overload"):
		return "overload"
	case strings.Contains(text, "过压") || strings.Contains(text, "欠压") || strings.Contains(text, "过流") || strings.Contains(text, "短路") || strings.Contains(text, "母线") || strings.Contains(text, "voltage") || strings.Contains(text, "current") || strings.Contains(text, "short circuit") || strings.Contains(text, "bus"):
		return "electrical"
	default:
		return "inverter"
	}
}

func metricForCategory(category, name string) (string, string, string) {
	switch category {
	case "thermal":
		return "temperature", "80", "C"
	case "battery":
		if strings.Contains(name, "欠压") {
			return "battery_voltage", "44", "V"
		}
		if strings.Contains(name, "过压") {
			return "battery_voltage", "58", "V"
		}
		return "battery_status", "1", "count"
	case "grid":
		if strings.Contains(name, "频率") {
			return "grid_frequency", "50", "Hz"
		}
		return "grid_voltage", "260", "V"
	case "cooling":
		return "fan_speed", "800", "rpm"
	case "communication":
		return "heartbeat_loss", "3", "count"
	case "pv":
		return "pv_voltage", "900", "V"
	case "overload":
		return "load_ratio", "100", "%"
	case "electrical":
		return "electrical_fault", "1", "count"
	case "safety":
		return "safety_check", "1", "count"
	default:
		return "fault_signal", "1", "count"
	}
}

func firstNonEmptyString(values ...string) string {
	for _, value := range values {
		if strings.TrimSpace(value) != "" {
			return value
		}
	}
	return ""
}

func metricValue(profile faultProfile, rng *rand.Rand) string {
	switch profile.Unit {
	case "C":
		return fmt.Sprintf("%.1f", 85+rng.Float64()*20)
	case "%":
		return fmt.Sprintf("%.1f", rng.Float64()*18)
	case "V":
		return fmt.Sprintf("%.1f", 760+rng.Float64()*80)
	case "rpm":
		return fmt.Sprintf("%d", rng.Intn(700))
	default:
		return fmt.Sprintf("%d", rng.Intn(10)+1)
	}
}

func eventTypeForMode(mode string, rng *rand.Rand) string {
	switch mode {
	case "resolved":
		return "resolved"
	case "mixed":
		n := rng.Intn(100)
		switch {
		case n < 5:
			return "resolved"
		case n < 20:
			return "update"
		default:
			return "firing"
		}
	default:
		return "firing"
	}
}

func extractAction(body []byte) string {
	var top map[string]json.RawMessage
	if err := json.Unmarshal(body, &top); err != nil {
		return ""
	}
	if raw, ok := top["action"]; ok {
		var action string
		_ = json.Unmarshal(raw, &action)
		return action
	}
	if raw, ok := top["data"]; ok {
		var data map[string]json.RawMessage
		if err := json.Unmarshal(raw, &data); err == nil {
			var action string
			_ = json.Unmarshal(data["action"], &action)
			return action
		}
	}
	return ""
}

func (s *stats) add(res result) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.total++
	if res.Error == "" {
		s.success++
	} else {
		s.failed++
		s.errors[res.Error]++
		if len(s.failSample) < 10 {
			s.failSample = append(s.failSample, res)
		}
	}
	if res.StatusCode > 0 {
		s.status[res.StatusCode]++
	}
	if res.Action != "" {
		s.actions[res.Action]++
	}
	if res.Latency > 0 {
		s.latencies = append(s.latencies, res.Latency)
	}
}

func printStats(s *stats, duration time.Duration) {
	fmt.Printf("Total: %d\n", s.total)
	fmt.Printf("Success: %d\n", s.success)
	fmt.Printf("Failed: %d\n", s.failed)
	fmt.Printf("Duration: %s\n", duration.Round(time.Millisecond))
	if duration > 0 {
		fmt.Printf("Throughput: %.2f req/s\n", float64(s.total)/duration.Seconds())
	}
	printIntMap("HTTP Status", s.status)
	printStringMap("Actions", s.actions)
	printLatency(s.latencies)
	printStringMap("Errors", s.errors)
	if len(s.failSample) > 0 {
		fmt.Println("\nFailure Samples:")
		for _, sample := range s.failSample {
			fmt.Printf("  seq=%d status=%d error=%s body=%s\n", sample.Sequence, sample.StatusCode, sample.Error, strings.TrimSpace(sample.Body))
		}
	}
}

func printIntMap(title string, values map[int]int) {
	if len(values) == 0 {
		return
	}
	fmt.Println("\n" + title + ":")
	keys := make([]int, 0, len(values))
	for key := range values {
		keys = append(keys, key)
	}
	sort.Ints(keys)
	for _, key := range keys {
		fmt.Printf("  %d: %d\n", key, values[key])
	}
}

func printStringMap(title string, values map[string]int) {
	if len(values) == 0 {
		return
	}
	fmt.Println("\n" + title + ":")
	keys := make([]string, 0, len(values))
	for key := range values {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	for _, key := range keys {
		fmt.Printf("  %s: %d\n", key, values[key])
	}
}

func printLatency(values []time.Duration) {
	if len(values) == 0 {
		return
	}
	sort.Slice(values, func(i, j int) bool { return values[i] < values[j] })
	total := time.Duration(0)
	for _, value := range values {
		total += value
	}
	fmt.Println("\nLatency:")
	fmt.Printf("  avg: %s\n", (total / time.Duration(len(values))).Round(time.Millisecond))
	fmt.Printf("  p50: %s\n", percentile(values, 50).Round(time.Millisecond))
	fmt.Printf("  p90: %s\n", percentile(values, 90).Round(time.Millisecond))
	fmt.Printf("  p99: %s\n", percentile(values, 99).Round(time.Millisecond))
	fmt.Printf("  max: %s\n", values[len(values)-1].Round(time.Millisecond))
}

func percentile(values []time.Duration, p float64) time.Duration {
	if len(values) == 0 {
		return 0
	}
	index := int(math.Ceil((p/100)*float64(len(values)))) - 1
	if index < 0 {
		index = 0
	}
	if index >= len(values) {
		index = len(values) - 1
	}
	return values[index]
}

func printSamplePayloads(cfg config) {
	limit := 3
	if cfg.Count < limit {
		limit = cfg.Count
	}
	for i := 1; i <= limit; i++ {
		body, _ := json.MarshalIndent(buildPayload(cfg, i), "", "  ")
		fmt.Printf("Sample %d:\n%s\n", i, string(body))
	}
}
