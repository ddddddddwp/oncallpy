# Local Stack Dependency Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the existing startup scripts restore Mem0, OnCall MCP, and authenticated Redis so the alarmmock chat flow runs without connection-refused dependency errors.

**Architecture:** Keep the two existing stack entry points. EnergyAgent's Python stack manager will load its ignored `.env` and validate Redis with an authenticated RESP `PING`; OnCallAgent's existing Bash restart script will start the existing Mem0 Compose project and existing MCP binary in dependency order. No new startup script or Mem0 source change is introduced.

**Tech Stack:** Python 3 standard library, pytest, Bash, Go, Docker Compose, Redis RESP, FastAPI health endpoints.

---

## File map

- Modify `/root/code/oncallpy/scripts/energyagent_stack.py`: `.env` precedence and authenticated Redis readiness.
- Modify `/root/code/oncallpy/tests/test_oncallpy_stack_script.py`: regression tests for URL precedence and authentication-aware dependency checks.
- Modify `/root/code/oncall/OnCallAgent/scripts/restart_local_stack.sh`: Mem0 and OnCall MCP lifecycle integration.
- Modify `/root/code/oncall/OnCallAgent/scripts/start_mcp.sh` only if its stale-PID behavior prevents reliable invocation from the main restart script.
- Modify `/root/code/oncallpy/.env`: local-only credential reconciliation; never stage or print this file.

### Task 1: Resolve Redis configuration from the project `.env`

**Files:**
- Modify: `/root/code/oncallpy/tests/test_oncallpy_stack_script.py`
- Modify: `/root/code/oncallpy/scripts/energyagent_stack.py`

- [ ] **Step 1: Add failing precedence tests**

Add tests that create a temporary project `.env`, clear ambient `REDIS_URL`, and verify this precedence:

```python
def test_stack_manager_reads_redis_url_from_project_dotenv(tmp_path, monkeypatch) -> None:
    module = load_stack_module()
    monkeypatch.delenv("REDIS_URL", raising=False)
    (tmp_path / ".env").write_text(
        "REDIS_URL=redis://:dotenv-secret@redis.local:6380/2\n",
        encoding="utf-8",
    )

    manager = module.StackManager(project_root=tmp_path, system=FakeSystem())

    assert manager.redis_url == "redis://:dotenv-secret@redis.local:6380/2"


def test_explicit_redis_url_overrides_environment_and_dotenv(tmp_path, monkeypatch) -> None:
    module = load_stack_module()
    monkeypatch.setenv("REDIS_URL", "redis://environment.local:6379/0")
    (tmp_path / ".env").write_text(
        "REDIS_URL=redis://dotenv.local:6379/0\n",
        encoding="utf-8",
    )

    manager = module.StackManager(
        project_root=tmp_path,
        system=FakeSystem(),
        redis_url="redis://explicit.local:6379/0",
    )

    assert manager.redis_url == "redis://explicit.local:6379/0"
```

- [ ] **Step 2: Run the new tests and confirm the old code fails**

Run:

```bash
.venv/bin/pytest -q \
  tests/test_oncallpy_stack_script.py::test_stack_manager_reads_redis_url_from_project_dotenv \
  tests/test_oncallpy_stack_script.py::test_explicit_redis_url_overrides_environment_and_dotenv
```

Expected: the `.env` test fails because `StackManager` currently only reads the explicit argument and process environment.

- [ ] **Step 3: Add a focused dotenv resolver**

Use the already installed `python-dotenv` package without importing the application settings layer:

```python
from dotenv import dotenv_values


def project_redis_url(project_root: Path) -> str | None:
    value = dotenv_values(project_root / ".env").get("REDIS_URL")
    return str(value).strip() if value else None
```

Resolve the URL in `StackManager.__init__`:

```python
self.redis_url = (
    redis_url
    or os.environ.get("REDIS_URL")
    or project_redis_url(project_root)
    or DEFAULT_REDIS_URL
)
```

- [ ] **Step 4: Run the complete stack-manager test file**

Run:

```bash
.venv/bin/pytest -q tests/test_oncallpy_stack_script.py
```

Expected: all tests pass.

### Task 2: Replace Redis TCP readiness with authenticated PING

**Files:**
- Modify: `/root/code/oncallpy/tests/test_oncallpy_stack_script.py`
- Modify: `/root/code/oncallpy/scripts/energyagent_stack.py`

- [ ] **Step 1: Change the fake boundary and add failing behavior tests**

Replace `FakeSystem.redis_port_open` with separate `redis_ready` and `redis_port_open` controls, then add:

```python
def test_open_redis_port_without_successful_ping_is_not_ready() -> None:
    module = load_stack_module()
    system = FakeSystem(redis_ready=False, redis_port_open=True)
    manager = module.StackManager(
        project_root=Path("/repo"),
        system=system,
        redis_url="redis://:wrong@localhost:6379/0",
    )

    with pytest.raises(TimeoutError, match="redis"):
        manager.ensure_dependencies()

    assert ("check", "redis-ping") in system.actions
    assert ("wait", "redis-ping") in system.actions


def test_ready_redis_uses_authenticated_url_without_exposing_it(capsys) -> None:
    module = load_stack_module()
    secret_url = "redis://:do-not-print@localhost:6379/0"
    system = FakeSystem(redis_ready=True, redis_port_open=True)
    manager = module.StackManager(
        project_root=Path("/repo"), system=system, redis_url=secret_url
    )

    manager.ensure_dependencies()

    assert secret_url not in capsys.readouterr().out
```

- [ ] **Step 2: Run the tests and confirm TCP-only behavior fails**

Run:

```bash
.venv/bin/pytest -q tests/test_oncallpy_stack_script.py -k redis
```

Expected: failures show that the manager only checks whether port 6379 accepts TCP.

- [ ] **Step 3: Implement a standard-library Redis PING**

Add private helpers that encode RESP arrays and read one Redis response without logging arguments:

```python
def _redis_command(*parts: str) -> bytes:
    encoded = [part.encode("utf-8") for part in parts]
    body = [f"*{len(encoded)}\r\n".encode()]
    for part in encoded:
        body.extend((f"${len(part)}\r\n".encode(), part, b"\r\n"))
    return b"".join(body)


def _read_redis_line(stream) -> bytes:
    response = stream.readline(4096)
    if not response.endswith(b"\r\n"):
        raise ConnectionError("incomplete Redis response")
    if response.startswith(b"-"):
        raise ConnectionError("Redis rejected the readiness command")
    return response
```

Implement `DockerSystem.redis_ready(redis_url)` using `urlparse`, `socket.create_connection`, optional `AUTH`, optional `SELECT`, and `PING`. Accept only `+OK` for authentication/selection and `+PONG` for readiness. The error path returns `False` and never includes the URL or password in output.

```python
def redis_ready(self, redis_url: str) -> bool:
    parsed = urlparse(redis_url)
    if parsed.scheme != "redis":
        return False
    host = parsed.hostname or "localhost"
    port = parsed.port or 6379
    username = unquote(parsed.username) if parsed.username else None
    password = unquote(parsed.password) if parsed.password else None
    database = parsed.path.lstrip("/") or "0"

    try:
        with socket.create_connection((host, port), timeout=1) as connection:
            stream = connection.makefile("rb")

            def execute(expected: bytes, *parts: str) -> bool:
                connection.sendall(_redis_command(*parts))
                return _read_redis_line(stream) == expected

            if password:
                auth = ("AUTH", username, password) if username else ("AUTH", password)
                if not execute(b"+OK\r\n", *auth):
                    return False
            if database != "0" and not execute(b"+OK\r\n", "SELECT", database):
                return False
            return execute(b"+PONG\r\n", "PING")
    except (ConnectionError, OSError, UnicodeError, ValueError):
        return False


def wait_for_redis(self, redis_url: str) -> None:
    self._wait_until(lambda: self.redis_ready(redis_url), "redis", timeout=60)
```

Update `StackManager.ensure_dependencies()`:

```python
if self.system.redis_ready(self.redis_url):
    print(f"redis: ready on {redis_host}:{redis_port}")
    return

if not self.system.redis_port_open(redis_host, redis_port):
    redis_containers = self.system.redis_container_names()
    if not redis_containers:
        raise RuntimeError(
            "redis: port is closed and no Docker container with name matching 'redis' was found"
        )
    container_name = redis_containers[0]
    print(f"redis: starting existing container {container_name}")
    self.system.start_container(container_name)
    self.system.wait_for_tcp(redis_host, redis_port, "redis")

self.system.wait_for_redis(self.redis_url)
```

- [ ] **Step 4: Run targeted and complete tests**

Run:

```bash
.venv/bin/pytest -q tests/test_oncallpy_stack_script.py
```

Expected: all stack-manager tests pass and captured output contains no Redis URL with credentials.

- [ ] **Step 5: Commit only the EnergyAgent stack changes**

Run:

```bash
git add scripts/energyagent_stack.py tests/test_oncallpy_stack_script.py
git commit -m "fix(stack): validate authenticated Redis readiness"
```

Do not stage `.env` or unrelated existing worktree changes.

### Task 3: Integrate Mem0 and OnCall MCP into the existing restart script

**Files:**
- Modify: `/root/code/oncall/OnCallAgent/scripts/restart_local_stack.sh`
- Modify if required: `/root/code/oncall/OnCallAgent/scripts/start_mcp.sh`

- [ ] **Step 1: Record the failing integration baseline**

With the current script running, verify the missing ports:

```bash
python3 - <<'PY'
import socket
for port in (8888, 12345):
    sock = socket.socket()
    sock.settimeout(1)
    try:
        sock.connect(("127.0.0.1", port))
        print(port, "reachable")
    except OSError:
        print(port, "unreachable")
    finally:
        sock.close()
PY
```

Expected before the fix: both ports report `unreachable`.

- [ ] **Step 2: Add Mem0 lifecycle functions to `restart_local_stack.sh`**

Add derived defaults and existing paths:

```bash
MEM0_STACK_DIR="${MEM0_STACK_DIR:-$(cd "$ROOT/../.." && pwd)/mem0/server}"
MEM0_COMPOSE_FILE="${MEM0_COMPOSE_FILE:-$MEM0_STACK_DIR/docker-compose.api.yaml}"
MEM0_URL="${MEM0_BASE_URL:-http://127.0.0.1:8888}"
```

Add `start_mem0` using the existing Compose file:

```bash
start_mem0() {
  [[ -f "$MEM0_COMPOSE_FILE" ]] || die "mem0 compose file not found: $MEM0_COMPOSE_FILE"
  log "mem0: starting postgres and api"
  docker compose -f "$MEM0_COMPOSE_FILE" up -d postgres mem0-api
  wait_for_http "${MEM0_URL%/}/docs" "mem0"
  log "mem0: ready url=$MEM0_URL"
}
```

Call `load_env` before `start_mem0`, and require `docker` when local Mem0 is configured.

- [ ] **Step 3: Add OnCall MCP stop/start functions to the same script**

Reuse `/tmp/oncall-mcp`, `logs/oncall_mcp.pid`, and `logs/oncall_mcp.log`. Stop stale PID, port, and process-pattern matches with the existing helpers. Build from `./internal/ai/cmd/oncall_mcp`, start after the backend is healthy, and call:

```bash
wait_for_port "12345" "oncall-mcp"
```

Keep `scripts/start_mcp.sh` as a supported standalone entry point; only adjust it if integration exposes stale PID handling that cannot be handled in `restart_local_stack.sh`.

- [ ] **Step 4: Put the existing components in dependency order**

The final `main` order is:

```text
load environment
stop backend/alarmcheck/alarmmock/MCP
start Mem0 Compose dependencies
start OnCall backend
start OnCall MCP
start alarmcheck
build alarmmock
```

- [ ] **Step 5: Validate shell syntax**

Run:

```bash
bash -n scripts/restart_local_stack.sh
bash -n scripts/start_mcp.sh
```

Expected: both commands exit 0.

- [ ] **Step 6: Restart the OnCall stack**

Run:

```bash
scripts/restart_local_stack.sh
```

Expected: Mem0 8888, OnCall 6872, OnCall MCP 12345, and alarmcheck 19090 all report ready; `/tmp/alarmmock` is rebuilt.

- [ ] **Step 7: Commit only the OnCallAgent script changes**

Run from `/root/code/oncall/OnCallAgent`:

```bash
git add scripts/restart_local_stack.sh scripts/start_mcp.sh
git commit -m "fix(stack): restore Mem0 and OnCall MCP"
```

If `start_mcp.sh` is unchanged, omit it from `git add`.

### Task 4: Permanently reconcile the ignored Redis configuration

**Files:**
- Modify: `/root/code/oncallpy/.env` (ignored, local only)

- [ ] **Step 1: Read the current container credential without printing it**

Inspect `1Panel-redis-VrI6` and extract the value following `--requirepass` entirely inside the tool process. Do not emit the value to stdout, logs, command arguments, or the plan.

- [ ] **Step 2: Update both dotenv keys atomically**

Set `REDIS_PASSWORD` to the extracted container value. Set `REDIS_URL` to a local database-0 URL whose password component is the percent-encoded form of the same value. Preserve all unrelated `.env` entries and apply the edit without surfacing the credential.

- [ ] **Step 3: Prove the file remains ignored and unstaged**

Run:

```bash
git check-ignore -v .env
git status --short -- .env
```

Expected: `.env` is ignored and absent from Git status.

- [ ] **Step 4: Restart EnergyAgent through the existing command**

Run:

```bash
scripts/oncallpy_stack.py restart
```

Expected: the stack manager reports authenticated Redis ready, both MCP services start, FastAPI becomes ready, and the command exits 0.

### Task 5: End-to-end verification and phase handoff

**Files:**
- No source changes expected.

- [ ] **Step 1: Verify every port and HTTP health endpoint**

Check ports 6379, 8000, 8003, 8004, 8888, 9900, 12345, 19090, 19530, and 6872. Verify:

```bash
curl -fsS http://127.0.0.1:8888/docs >/dev/null
curl -fsS http://127.0.0.1:9900/health
curl -fsS http://127.0.0.1:6872/api/alerts >/dev/null
```

Expected: all commands exit 0; EnergyAgent reports Milvus and Redis status `ok`.

- [ ] **Step 2: Run repository-level tests**

Run in `/root/code/oncallpy`:

```bash
.venv/bin/pytest -q tests/test_oncallpy_stack_script.py tests/test_health_components.py
```

Run in `/root/code/oncall/OnCallAgent`:

```bash
go test ./...
```

Expected: all tests pass.

- [ ] **Step 3: Inject the requested alert again**

Run:

```bash
/tmp/alarmmock -url http://127.0.0.1:6872/api/alerts/ingest \
  -count 1 -concurrency 1 \
  -fault-code E_VOLT_HIGH \
  -fault-name "英威腾逆变器电压过高" \
  -fault-category electrical \
  -same-alert
```

Expected: `Success: 1`, `Failed: 0`, HTTP 200.

- [ ] **Step 4: Re-run the ten-question INVT session with a new session ID**

POST the same ten questions from the approved alarmmock scenario to the existing 6872 chat endpoint in order. Require ten successful responses, `vendor-invt-register-list-20260529` retrieval, and zero HS/BST document retrievals.

- [ ] **Step 5: Verify dependency errors are gone for the new session**

Parse `/root/code/oncall/OnCallAgent/logs/backend.log` for the new session ID. Expected:

```text
chat_request_end status=ok: 10
127.0.0.1:8888 connection refused: 0
127.0.0.1:12345 connection refused: 0
vendor-invt-register-list-20260529 retrieval rounds: 10
HS/BST retrieval rounds: 0
```

- [ ] **Step 6: Begin phase 2 only after all phase 1 checks pass**

Use CodeGraph to inventory `/root/code/oncallpy`, classify cleanup candidates as proven-unused, duplicate, generated/runtime, or intentionally retained, and present the deletion batches before removing files.
