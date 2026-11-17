# EnergyAgent Directory Cutover and Main Merge Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `/root/code/EnergyAgent` the canonical local repository, preserve `/root/code/oncallpy` compatibility, integrate the reviewed working-tree code, and fast-forward local `main` after all tests pass.

**Architecture:** Keep `app` and every existing import/command compatibility shim stable. Convert the dirty working tree into small tested commits, stop the three EnergyAgent processes, move the directory, create a relative compatibility link, restart through the canonical command, and only then fast-forward `main`. Generated files, local credentials, and nonexistent implementations stay out of Git.

**Tech Stack:** Python 3.10+, FastAPI, Pydantic, pytest, Ruff, Go, Git, Docker, Redis, Milvus, Bash, symbolic links.

---

## File Map

- `.gitignore`: ignore local workflow state, live logs, and the compiled alarmmock binary.
- `docs/ONBOARDING.md`, `tests/test_energyagent_branding.py`: finish maintained EnergyAgent branding.
- `app/main.py`, `tests/test_app_startup.py`: register and verify the migrated on-call router.
- `app/api/oncall.py`, `app/domains/oncall/`, `tests/test_oncall_migration.py`: make the on-call migration reachable, deterministic, and tested.
- `app/domains/alerts/`: expose available alert contracts without importing a nonexistent store.
- `app/models/aiops.py`: remove one proven unused import.
- `app/core/config_migration.py`, `scripts/migrate_oncall_config.py`, `tests/test_config_migration.py`: commit reviewed Redis reconciliation.
- `app/config.py`, `app/rag/milvus_client.py`, `app/rag/vector_store_manager.py` and their tests: commit reviewed Milvus database selection.
- `alarmcheck/`, `alarmmock/`: commit Go source, tests, modules, and documentation without an ELF binary.

### Task 1: Establish the Safety Baseline

**Files:** Inspect only.

- [ ] **Step 1: Prove destination and Git preconditions**

Run:

```bash
test ! -e /root/code/EnergyAgent
test ! -L /root/code/EnergyAgent
git branch --show-current
git merge-base --is-ancestor main HEAD
git rev-list --left-right --count main...HEAD
```

Expected: both path checks and the ancestor check exit 0; branch is `checkpoint/energyagent-pre-migration`; the left count is `0`.

- [ ] **Step 2: Record worktree state and run the focused baseline**

Run:

```bash
git status --short
git worktree list --porcelain
git diff --stat
git diff --cached --stat
.venv/bin/pytest -q tests/test_energyagent_branding.py tests/test_oncallpy_stack_script.py tests/test_health_components.py
```

Expected: known migration files only, existing linked worktrees remain registered, and tests exit 0.

### Task 2: Exclude Generated State and Finish Branding

**Files:**
- Modify: `.gitignore`
- Modify: `docs/ONBOARDING.md`
- Modify: `tests/test_energyagent_branding.py`
- Index-only removal: `alarmmock/alarmmock`

- [ ] **Step 1: Add a failing maintained-document branding assertion**

Add this entry to the `paths` list in `test_product_surfaces_do_not_present_legacy_branding`:

```python
        ROOT / "docs/ONBOARDING.md",
```

Run:

```bash
.venv/bin/pytest -q tests/test_energyagent_branding.py::test_product_surfaces_do_not_present_legacy_branding
```

Expected: FAIL because the onboarding document still presents `oncallpy` as the product name.

- [ ] **Step 2: Update the three maintained onboarding references**

Use these canonical sentences while preserving OnCall business terminology:

```markdown
# EnergyAgent 项目学习文档

本文用于帮助新加入项目的人快速理解 EnergyAgent 的整体结构、运行方式和主要业务流程。文档只讲架构、模块职责和数据流，不包含具体源码片段。

EnergyAgent 是一个面向智能 OnCall 场景的后端应用，核心能力是：
```

Change the concluding product reference to:

```markdown
EnergyAgent 的核心不是单纯聊天，而是“知识库 + 工具调用 + 运维诊断工作流”的组合。
```

- [ ] **Step 3: Add explicit local-state ignore rules**

Append without removing the existing user rules:

```gitignore

# Local workflow and runtime state
.comet/
/runtime-logs/
/alarmmock/alarmmock
```

Remove only the compiled binary from the index and verify ignores:

```bash
git rm --cached -f -- alarmmock/alarmmock
git check-ignore -v .comet/config.yaml runtime-logs/server.log alarmmock/alarmmock
```

Expected: all three paths are ignored; Go source remains.

- [ ] **Step 4: Run and commit the hygiene batch**

Run:

```bash
.venv/bin/pytest -q tests/test_energyagent_branding.py
git add .gitignore tests/test_energyagent_branding.py
git add -f docs/ONBOARDING.md
git commit --only -m "chore: complete EnergyAgent repository hygiene" -- .gitignore docs/ONBOARDING.md tests/test_energyagent_branding.py
```

Expected: tests pass and only the three listed files are committed.

### Task 3: Remove Proven Static-Analysis Debris

**Files:**
- Modify: `app/models/aiops.py`
- Modify with Task 4: `app/api/oncall.py`
- Modify with Task 4: `app/domains/oncall/store.py`
- Modify with Task 4: `app/domains/oncall/utils.py`

- [ ] **Step 1: Capture the failing lint baseline**

Run:

```bash
.venv/bin/ruff check app tests --select F401,F811,F841
```

Expected: FAIL only for the audited unused imports, redundant aliases, and duplicate on-call helpers.

- [ ] **Step 2: Remove the unused AIOps `List` import**

Replace:

```python
from typing import Optional, List, Dict, Any
```

with:

```python
from typing import Any, Dict, Optional
```

Run and commit:

```bash
.venv/bin/ruff check app/models/aiops.py --select F401,F811,F841
git add app/models/aiops.py
git commit --only -m "chore: remove unused AIOps import" -- app/models/aiops.py
```

Expected: Ruff passes and only `app/models/aiops.py` is committed.

### Task 4: Integrate the Alert and On-Call Migration Slice

**Files:**
- Create: `tests/test_oncall_migration.py`
- Modify: `tests/test_app_startup.py`
- Modify: `app/main.py`
- Modify: `app/api/oncall.py`
- Add/modify: `app/domains/alerts/`
- Add/modify: `app/domains/oncall/`

- [ ] **Step 1: Write failing migration contract tests**

Create `tests/test_oncall_migration.py`:

```python
from datetime import timezone

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api import oncall
from app.domains.alerts.models import ExternalAlertPayload
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
```

- [ ] **Step 2: Add a failing application-router test**

In `_load_isolated_main`, include `oncall` in the fake API module loop and add a probe route:

```python
    for name in ("aiops", "chat", "file", "health", "oncall"):
        module = types.ModuleType(f"app.api.{name}")
        module.router = APIRouter()
        if name == "oncall":
            module.router.add_api_route(
                "/oncall-probe",
                lambda: {"status": "ok"},
                methods=["GET"],
            )
        setattr(fake_api, name, module)
        monkeypatch.setitem(sys.modules, module.__name__, module)
```

Add:

```python
def test_main_registers_oncall_routes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    main_module = _load_isolated_main(tmp_path, monkeypatch)
    assert "/api/oncall-probe" in {route.path for route in main_module.app.routes}
```

Run:

```bash
.venv/bin/pytest -q tests/test_oncall_migration.py tests/test_app_startup.py::test_main_registers_oncall_routes
```

Expected: FAIL for the missing alert store import, undefined UTC name, duplicate aggregation, and absent router registration.

- [ ] **Step 3: Export only available alert contracts**

Replace `app/domains/alerts/__init__.py` with:

```python
"""Alert domain contracts available during the migration release."""

from .errors import (
    AlertError,
    AlertNotFound,
    AlertSilenceError,
    AlertValidationError,
    InvalidAlertState,
)
from .models import (
    ALERT_STATES,
    Alert,
    AlertEvent,
    AlertIngestInput,
    AlertIngestResult,
    ExternalAlertPayload,
    Note,
    PrometheusAlert,
    PrometheusAlertsResult,
    SilenceMatcher,
    SilenceRule,
    TimelineEvent,
)

__all__ = [
    "ALERT_STATES", "Alert", "AlertError", "AlertEvent", "AlertIngestInput",
    "AlertIngestResult", "AlertNotFound", "AlertSilenceError",
    "AlertValidationError", "ExternalAlertPayload", "InvalidAlertState", "Note",
    "PrometheusAlert", "PrometheusAlertsResult", "SilenceMatcher", "SilenceRule",
    "TimelineEvent",
]
```

- [ ] **Step 4: Register the source-compatible routes**

Change the import and add the router in `app/main.py`:

```python
from app.api import aiops, chat, file, health, oncall
```

```python
app.include_router(oncall.router, prefix="/api", tags=["值班管理"])
```

The source Go application also uses `/api`, preserving `/api/escalation-policies` and `/api/oncall/{team}`.

- [ ] **Step 5: Remove exact on-call duplication and undefined/unused names**

In `app/api/oncall.py`, remove `OnCallState`, the five `_Unused...` aliases, and `zoneinfo`. Keep the typed imports used in return annotations. Reduce aggregation to:

```python
        result = _store.aggregate_alert(alert)
        return {"result": result}
```

In `app/domains/oncall/store.py`, remove only `compact_first_non_empty` from the `.utils` import list.

In `app/domains/oncall/utils.py`, delete the earlier `max_time` and `min_time` definitions so the effective later versions remain. Replace `now_utc` with:

```python
def now_utc() -> datetime:
    return datetime.now(timezone.utc)
```

- [ ] **Step 6: Verify and commit the slice**

Run:

```bash
.venv/bin/pytest -q tests/test_oncall_migration.py tests/test_app_startup.py
.venv/bin/ruff check app/api/oncall.py app/domains/alerts app/domains/oncall tests/test_oncall_migration.py tests/test_app_startup.py --select F401,F811,F841
git add app/main.py app/api/oncall.py app/domains/alerts app/domains/oncall tests/test_app_startup.py tests/test_oncall_migration.py
git commit --only -m "feat: integrate alert and on-call migration contracts" -- app/main.py app/api/oncall.py app/domains/alerts app/domains/oncall tests/test_app_startup.py tests/test_oncall_migration.py
```

Expected: tests and Ruff pass; the commit contains reachable on-call APIs, import-safe alert contracts, and no runtime state.

### Task 5: Commit Redis Reconciliation

**Files:** `app/core/config_migration.py`, `scripts/migrate_oncall_config.py`, `tests/test_config_migration.py`.

- [ ] **Step 1: Verify the existing implementation**

Run:

```bash
.venv/bin/pytest -q tests/test_config_migration.py
.venv/bin/ruff check app/core/config_migration.py scripts/migrate_oncall_config.py tests/test_config_migration.py --select F401,F811,F841
```

Expected: exit 0 with fake credentials only.

- [ ] **Step 2: Commit exactly the batch**

Run:

```bash
git add app/core/config_migration.py scripts/migrate_oncall_config.py tests/test_config_migration.py
git commit --only -m "fix(config): reconcile authenticated Redis URLs" -- app/core/config_migration.py scripts/migrate_oncall_config.py tests/test_config_migration.py
```

Expected: only the three listed files are committed.

### Task 6: Commit Milvus Database Selection

**Files:** `app/config.py`, `app/rag/milvus_client.py`, `app/rag/vector_store_manager.py`, `tests/test_layered_config.py`, `tests/test_milvus_database_selection.py`, `tests/test_vector_store_filtering.py`.

- [ ] **Step 1: Verify the existing implementation**

Run:

```bash
.venv/bin/pytest -q tests/test_layered_config.py tests/test_milvus_database_selection.py tests/test_vector_store_filtering.py
.venv/bin/ruff check app/config.py app/rag/milvus_client.py app/rag/vector_store_manager.py tests/test_layered_config.py tests/test_milvus_database_selection.py tests/test_vector_store_filtering.py --select F401,F811,F841
```

Expected: tests prove native and LangChain clients select the configured database; Ruff exits 0.

- [ ] **Step 2: Commit exactly the batch**

Run:

```bash
git add app/config.py app/rag/milvus_client.py app/rag/vector_store_manager.py tests/test_layered_config.py tests/test_milvus_database_selection.py tests/test_vector_store_filtering.py
git commit --only -m "fix(rag): select the configured Milvus database" -- app/config.py app/rag/milvus_client.py app/rag/vector_store_manager.py tests/test_layered_config.py tests/test_milvus_database_selection.py tests/test_vector_store_filtering.py
```

Expected: only the six listed files are committed.

### Task 7: Validate and Commit Alarm Tools

**Files:** Go source, tests, modules, and `alarmcheck/README.md`; exclude `alarmmock/alarmmock`.

- [ ] **Step 1: Format and test both modules**

Run:

```bash
gofmt -w alarmcheck/alarmcheck.go alarmcheck/alarmcheck_test.go alarmmock/alarmmock.go alarmmock/alarmmock_test.go
go -C alarmcheck test ./...
go -C alarmmock test ./...
```

Expected: both suites pass.

- [ ] **Step 2: Commit source only**

Run:

```bash
git add alarmcheck/README.md alarmcheck/alarmcheck.go alarmcheck/alarmcheck_test.go alarmcheck/go.mod alarmmock/alarmmock.go alarmmock/alarmmock_test.go alarmmock/go.mod
git commit --only -m "feat(tools): add alarm mock and verification utilities" -- alarmcheck/README.md alarmcheck/alarmcheck.go alarmcheck/alarmcheck_test.go alarmcheck/go.mod alarmmock/alarmmock.go alarmmock/alarmmock_test.go alarmmock/go.mod
git show --stat --oneline HEAD
```

Expected: no ELF binary appears in the commit.

### Task 8: Complete Pre-Cutover Verification

**Files:** Verify only.

- [ ] **Step 1: Require a clean visible tree and static checks**

Run:

```bash
git status --short
git diff --check
git diff --cached --check
.venv/bin/ruff check app tests --select F401,F811,F841
.venv/bin/python -m compileall -q app scripts tests
```

Expected: status is empty and every command exits 0. Ignored `.env`, `.comet/`, `runtime-logs/`, and the local binary may remain on disk.

- [ ] **Step 2: Run complete Python and Go tests**

Run:

```bash
.venv/bin/pytest -q
go -C alarmcheck test ./...
go -C alarmmock test ./...
```

Expected: every suite exits 0.

- [ ] **Step 3: Scan main-to-HEAD content**

Run:

```bash
git diff --name-only main...HEAD
git diff --numstat main...HEAD | sort -nr | head -30
git grep -n -E 'AKIA[0-9A-Z]{16}|sk-[A-Za-z0-9_-]{20,}|BEGIN (RSA |EC |OPENSSH )?PRIVATE KEY' HEAD -- . ':!tests'
```

Expected: no compiled binary and no high-confidence secret match.

### Task 9: Move the Repository and Restore the Stack

**Files:** Move `/root/code/oncallpy` to `/root/code/EnergyAgent`; create `/root/code/oncallpy -> EnergyAgent`.

- [ ] **Step 1: Recheck and stop only EnergyAgent-owned Python services**

Run from `/root/code/oncallpy`:

```bash
test ! -e /root/code/EnergyAgent
test ! -L /root/code/EnergyAgent
git status --short
.venv/bin/python -c "from pathlib import Path; from scripts.energyagent_stack import StackManager; manager = StackManager(project_root=Path.cwd()); [manager.system.stop_service(service) for service in manager.services]"
```

Expected: clean tree; FastAPI and ports 8003/8004 stop; Docker, Redis, Milvus, Mem0, OnCallAgent, and alarmcheck remain running.

- [ ] **Step 2: Move and link**

Run from `/root/code`:

```bash
mv -- oncallpy EnergyAgent
ln -s -- EnergyAgent oncallpy
```

Expected: `EnergyAgent` is a real directory and `oncallpy` is a relative link.

- [ ] **Step 3: Validate both paths and linked worktrees**

Run:

```bash
realpath /root/code/EnergyAgent
realpath /root/code/oncallpy
git -C /root/code/EnergyAgent rev-parse --show-toplevel
git -C /root/code/oncallpy rev-parse --show-toplevel
git -C /root/code/EnergyAgent worktree list --porcelain
```

Expected: both paths resolve to `/root/code/EnergyAgent` and all linked worktrees remain registered.

- [ ] **Step 4: Restart canonically and check health**

Run:

```bash
cd /root/code/EnergyAgent
scripts/energyagent_stack.py restart
curl -fsS http://127.0.0.1:9900/health
```

Expected: exit 0 and healthy Redis, Milvus, and configuration.

- [ ] **Step 5: Roll back only if path or restart validation fails**

Before `main` changes, run only on failure:

```bash
cd /root/code
rm -- oncallpy
mv -- EnergyAgent oncallpy
```

Expected: original physical directory restored; diagnose before retrying.

### Task 10: Fast-Forward Main and Verify

**Files:** Update local `main`; retain checkpoint and foundation branches.

- [ ] **Step 1: Prove and perform the fast-forward**

Run:

```bash
cd /root/code/EnergyAgent
git merge-base --is-ancestor main checkpoint/energyagent-pre-migration
git status --short
git switch main
git merge --ff-only checkpoint/energyagent-pre-migration
```

Expected: clean tree and fast-forward; no merge commit, pull, push, or remote rename.

- [ ] **Step 2: Repeat static and automated verification on main**

Run:

```bash
.venv/bin/ruff check app tests --select F401,F811,F841
.venv/bin/pytest -q
go -C alarmcheck test ./...
go -C alarmmock test ./...
```

Expected: every command exits 0.

- [ ] **Step 3: Run the post-merge alarm smoke path**

Run:

```bash
cd /root/code/EnergyAgent/alarmmock
go build -o /tmp/alarmmock .
/tmp/alarmmock -url http://127.0.0.1:6872/api/alerts/ingest -count 1 -concurrency 1 -fault-code E_VOLT_HIGH -fault-name "英威腾逆变器电压过高" -fault-category electrical -same-alert
```

Check listeners and health:

```bash
cd /root/code/EnergyAgent
.venv/bin/python -c "import socket; ports = (6379, 8000, 8003, 8004, 8888, 9900, 12345, 19090, 19530, 6872); connections = [socket.create_connection(('127.0.0.1', port), 2) for port in ports]; [connection.close() for connection in connections]; print('ports_ok', len(ports))"
curl -fsS http://127.0.0.1:9900/health
```

Expected: alarmmock reports one success and zero failures; all ten listeners and health succeed.

- [ ] **Step 4: Confirm final state**

Run:

```bash
git branch --show-current
git status --short
git branch --list main checkpoint/energyagent-pre-migration feat/energyagent-foundation
readlink /root/code/oncallpy
git log --oneline --decorate -8
```

Expected: current branch `main`, clean status, retained rollback branches, link target `EnergyAgent`, and integrated commits at the tip of `main`.
