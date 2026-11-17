---
role: technical-design
program: energyagent-native-migration
source_repository: /root/code/oncall
source_commit: d9209560e5fea977b9aff59d9b02f836580f6019
target_repository: /root/code/oncallpy
target_commit: aa74b1450899fac2d65cd3bdcd05b4ceca5b5742
decision: python-native-port
---

# EnergyAgent Python-Native Migration Design

## Goal

Rename `oncallpy` to **EnergyAgent** and migrate the implemented product capabilities that exist in `/root/code/oncall` but do not exist in the Python project. EnergyAgent remains a FastAPI, Pydantic, LangGraph, Redis, and Milvus application. The migration ports behavior, API contracts, persisted state, operational tooling, and local runtime configuration; it does not copy the Go implementation into the target or replace the Python RAG stack.

The completed system must preserve all current Python behavior while closing the source capability gap. The source baseline is commit `d9209560e5fea977b9aff59d9b02f836580f6019`. The target baseline is commit `aa74b1450899fac2d65cd3bdcd05b4ceca5b5742` plus the existing uncommitted RAG refactor in the working tree.

## Decision

EnergyAgent uses one production runtime: Python.

- FastAPI owns HTTP and SSE contracts.
- Pydantic owns request, response, domain, and configuration validation.
- LangGraph owns chat and AIOps orchestration.
- Redis remains the conversation/checkpoint store.
- Milvus plus the current sparse index remains the RAG retrieval stack.
- New alert, user, on-call, silence, and evaluation state is accessed through repository interfaces.
- Source Go code is reference behavior and test evidence, not a runtime dependency.

This decision rejects two alternatives:

1. Running the Go service beside FastAPI would shorten initial delivery but leave two configuration systems, two observability stacks, and duplicated domain ownership.
2. Replacing the Python project with the Go project would discard the current RAG, Redis checkpoint, MCP, and multi-agent work.

## Current Baseline and Constraints

### Target strengths that must be retained

- Non-streaming and streaming RAG chat.
- Redis-backed LangGraph checkpoints and session operations.
- Single-agent and multi-agent AIOps execution.
- MCP clients and local CLS/monitor MCP servers.
- Milvus vector retrieval, sparse retrieval, rank fusion, metadata routing, protocol metadata, ingestion deduplication, and the current RAG refactor.
- FastAPI static frontend and local stack management.

### Source-only capabilities to migrate

- Alert ingest, Prometheus synchronization, list/detail, workflow state, notes, event history, claim, reassign, release, and silence.
- Silence rule CRUD and label/annotation matching.
- Escalation policies, manual escalation, alert aggregation, schedules, shifts, current on-call calculation, routing rules, rule matching, and automatic assignment.
- Local users, authentication sessions, roles, password policy, and user administration.
- Single-alert root-cause analysis, automatic remediation, solution-code resolution, cloud command dispatch, and audit trails.
- Long-term Mem0 memory behavior not already provided by Redis checkpoints.
- RAG evaluation datasets, runs, scoring, API, CLI, history, and dashboard.
- Application container packaging, Redis compose ownership, full observability deployment, structured log investigation, and end-to-end alert mock/check tooling.

### Working-tree protection

The target contains substantial uncommitted RAG work. No implementation worktree may be created from `HEAD` alone because it would omit that work. Before implementation, the current state must be protected by one of these user-authorized mechanisms:

1. a dedicated checkpoint commit containing the existing RAG work, or
2. an immutable patch bundle plus untracked-file archive followed by a verified restore rehearsal.

The preferred mechanism is a checkpoint commit after the existing RAG tests pass. Secrets, logs, databases, uploads, runtime volumes, and generated Graphify output are excluded from that commit.

## Non-Goals

- Do not embed or invoke the source Go application from production EnergyAgent.
- Do not rewrite the current RAG pipeline merely to resemble the source implementation.
- Do not mechanically replace every occurrence of `oncall`; on-call scheduling is a domain term.
- Do not commit keys, tokens, passwords, DSNs, or local secret snapshots.
- Do not migrate IDE state, caches, logs, PID files as durable data, local databases, uploads, or Milvus volumes by blind directory copying.
- Do not preserve request-path process termination behavior such as `log.Fatal`; migrated code returns typed errors.
- Do not enable live remediation solely because configuration was imported. Live commands require an explicit runtime mode and authorization.

## Program Decomposition

This program is too large for one implementation plan. It is divided into independently verifiable delivery packages:

1. **Foundation:** baseline protection, EnergyAgent naming, configuration, secret migration, compatibility aliases, repositories, infrastructure, and health contracts.
2. **Alert Operations:** alert lifecycle, ingest, Prometheus synchronization, event history, ownership, notes, and silences.
3. **Users and On-Call:** authentication, roles, escalation, aggregation, schedules, routing, and automatic assignment.
4. **Remediation:** single-alert analysis, solution resolution, dry-run/live command dispatch, idempotency, authorization, and audit.
5. **AI Memory and RAG Evaluation:** Mem0 integration, resilience parity, evaluation service, APIs, CLI, datasets, and dashboard.
6. **Frontend, Observability, and Cutover:** workbench UI, operational stack, mock/check harnesses, compatibility removal policy, physical repository rename, and final parity validation.

Every package receives its own implementation plan and produces a deployable, testable increment. Foundation is first. Alert Operations and Users and On-Call may proceed in parallel after their shared repository and identity contracts are stable. Remediation depends on Alert Operations. Frontend integration follows stable APIs but can develop against contract fixtures.

## Naming and Compatibility Design

### Canonical names

| Surface | Canonical name |
|---|---|
| Product and UI | `EnergyAgent` |
| Python distribution | `energy-agent` |
| Service slug and container prefix | `energyagent` |
| Log/trace service identity | `energy-agent` |
| Browser storage prefix | `energyagent` |
| Final repository directory | `/root/code/EnergyAgent` |

The internal Python package remains `app` during this program. Renaming it provides no product value and would create a large import-only change.

### Compatibility period

Compatibility lasts through the migration release and at least one subsequent release:

- New environment variables are read first, legacy variables second, defaults last.
- Legacy `SUPER_BIZ_AGENT_*` and applicable `ONCALL_*` variables emit a deprecation warning that contains the key name but never its value.
- New state paths are preferred; legacy `/tmp/super_biz_agent/...` state is copied and checksum-verified before use.
- New browser keys are preferred; `superbizagent.opsManagement` is read once, migrated to the EnergyAgent key, and retained as rollback evidence.
- Observability dashboards query both `energy-agent` and legacy `oncall-agent` identities during the compatibility period.
- Renamed scripts keep small forwarding shims for their old filenames for one release.

The physical repository directory is renamed only in the final cutover package, after scripts, documentation, tests, and local tooling no longer depend on `/root/code/oncallpy`.

## Configuration and Secret Migration

### Configuration layers

EnergyAgent uses three explicit configuration layers, in increasing precedence:

1. tracked, non-secret `config/energyagent.yaml` defaults;
2. ignored `config/local.yaml` machine and deployment overrides;
3. ignored `.env` secrets and emergency environment overrides.

Pydantic Settings exposes one typed configuration object to the application. Domain services receive only their configuration subsection rather than importing global settings throughout the codebase.

### Complete local preservation

Before transforming configuration, copy every in-scope source runtime configuration file into an ignored migration vault:

```text
.energyagent/migration/<timestamp>/
  source-manifest.json
  source/OnCallAgent/.env.local
  source/OnCallAgent/.env.example
  source/OnCallAgent/config.yaml
  source/OnCallAgent/manifest/docker/docker-compose.yml
  source/OnCallAgent/manifest/observability/...
  target-before/.env
  target-before/config-local-if-present
```

The vault directory is mode `0700`; files containing potential secrets are mode `0600`. `source-manifest.json` stores paths, modes, sizes, key names, and SHA-256 fingerprints, never secret values. The vault is excluded from Git before it is created.

Developer-tool files are handled separately:

- Identical `.codex/settings.local.json` and `.comet/config.yaml` are recorded as already satisfied rather than duplicated.
- `.comet/config.yaml` becomes either explicitly tracked or ignored; it cannot remain untracked and unignored.
- IDE workspace state is not migrated.

### Key-level transformation

- Preserve every target-only key.
- Preserve every source-only key in the vault even before its consumer is implemented.
- Map `MILVUS_ADDRESS` into typed host and port fields while retaining the original source key in the compatibility profile.
- Map nested source YAML fields into typed EnergyAgent sections for admin, API limits, CORS, database, model providers, uploads, JWT, MCP, Mem0, Milvus, Prometheus, RAG, server, and state paths.
- Unknown source keys are retained in the migration manifest and cause a review warning; they are never silently dropped.

The conflicting `COZELOOP_API_TOKEN` and `DEEPSEEK_API_KEY` values coexist as two named provider profiles:

- `current` retains the existing target value and remains the default for existing EnergyAgent behavior.
- `oncall_legacy` retains the imported source value and is used by newly ported behavior until parity tests prove that the profiles can be unified.

The values are compared only by fingerprint. No command, test output, report, exception, or API response may display them.

### Atomic migration

Configuration migration writes same-filesystem temporary files under a restrictive umask, validates them, sets mode `0600`, and atomically renames them into place. Before replacement it creates a mode-`0600` local backup. Validation checks:

- required keys are present and non-empty;
- target-only keys remain present;
- key fingerprints match the approved profile mapping;
- `.env`, `config/local.yaml`, the vault, and backups are ignored and untracked;
- application startup and health checks succeed without dumping the environment.

Tracked source configuration containing real credentials is treated as compromised. EnergyAgent removes those values from tracked templates, reports which provider keys require rotation, and waits for external credential owners to rotate them. Git-history rewriting is a separate explicitly authorized operation.

## Domain Architecture

New capabilities are organized by domain instead of being added to the existing RAG service layer.

```text
app/
  domains/
    alerts/
    identity/
    oncall/
    remediation/
    rag_eval/
  integrations/
    prometheus/
    cloud_command/
    mem0/
    observability/
  api/
  rag/
```

Each domain exposes application services and repository protocols. FastAPI routes depend on application services. Domain code does not depend on FastAPI, Redis clients, filesystem APIs, or provider SDKs directly.

### Alerts

The alert aggregate owns:

- source identity and deduplication identity;
- labels, annotations, severity, device and fault context;
- workflow state, owner, notes, and timeline;
- silence and escalation status;
- remediation and cloud-command references.

Alert ingest is idempotent by `(source, source_event_id)`. Prometheus synchronization merges active alerts without erasing user-owned workflow fields. Alerts absent from Prometheus are resolved or removed using the same touched/untouched behavior as the source.

Local state initially uses atomic JSON repositories so existing source state can be migrated without a schema conversion. Optional MySQL alert-ingest persistence is implemented behind the same repository protocol when a database DSN is configured. Repository writes use temporary-file plus rename semantics or database transactions.

### Identity

Identity provides users, roles, password policy, login, logout, current-user lookup, and administration. Existing bcrypt hashes remain verifiable during migration; new passwords use Argon2id, and a successful login transparently upgrades a bcrypt hash to Argon2id. Authentication sessions use Redis with expiry; user records remain in an atomic local state repository during initial parity. Authorization is enforced in application services, not only in route decorators.

### On-Call

The on-call domain owns escalation policies, aggregation rules, schedules, layers, rotations, time-zone restrictions, routing rules, current-on-call calculation, and automatic assignment. It references user IDs through an identity service interface and alert IDs through an alert assignment interface; it does not import either domain's persistence adapter.

### Remediation

Remediation resolves a fault code to a solution code using the existing Python RAG interface, maps the result to an allowlisted command, sends through a cloud-command gateway, and writes an immutable audit event.

Runtime modes are explicit:

- `disabled`: no commands are generated or sent;
- `dry_run`: commands are resolved and audited but not sent; this is the default;
- `live`: commands may be sent only for authorized users and allowlisted device/command combinations.

Every request carries an idempotency key. Retries cannot send the same live command twice. Timeout, provider failure, invalid knowledge evidence, denied authorization, and unsupported command all return typed failures without terminating the process.

### RAG Evaluation and Memory

RAG evaluation calls `RagRetrievalService` and the existing answer pipeline through stable interfaces. It stores datasets, cases, runs, per-case evidence, metrics, and aggregate summaries independently from production chat history.

Mem0 supplements Redis checkpoints:

- Redis remains authoritative for conversation execution and recent messages.
- Mem0 provides optional cross-session long-term memory.
- Search failure degrades to Redis-only behavior.
- Auto-write is configurable and never blocks the response path.
- Memory scope is derived from the authenticated user when present and from the conversation/session compatibility ID otherwise.

## API Compatibility

The existing eight Python routes remain available and keep their existing request aliases and streaming behavior:

- `/api/chat`
- `/api/chat_stream`
- `/api/chat/clear`
- `/api/chat/session/{session_id}`
- `/api/aiops`
- `/api/upload`
- `/api/index_directory`
- `/api/health`

The source parity ledger covers the remaining route families:

- alert list/detail/state/notes/analysis;
- alert ingest/events/claim/reassign/release/silence/auto-remediate/escalate;
- silence CRUD;
- escalation policy CRUD;
- alert groups and aggregation rule CRUD;
- schedule CRUD, shifts, current on-call, and team on-call;
- routing rule CRUD and matching;
- identity endpoints for login, logout, current user, user list/create/update/disable, and password reset, preserving the authorization rules implemented by the source user domain;
- fault command send;
- RAG evaluation cases, run creation, run list, and run detail.

All new non-streaming routes use the EnergyAgent envelope:

```json
{
  "code": 200,
  "message": "success",
  "data": {}
}
```

HTTP status and envelope `code` agree. Adding `code` remains compatible with source clients that ignore unknown fields. SSE routes bypass the envelope and use the existing Python event contract. Request models accept the source field casing as aliases where source clients require it, but generated OpenAPI presents canonical snake_case names.

The parity ledger records, for every source route: method, path, request fields, response fields, errors, authorization, persistence effects, frontend consumer, migrated test, and final status. A route is not complete until its contract and side effects pass.

## Data Migration

Data migration is copy-first and reversible:

1. Stop writers or enter maintenance mode.
2. Snapshot source and target state with paths, modes, sizes, counts, and checksums.
3. Copy source alert, silence, user, on-call, routing, and RAG-evaluation state into staging.
4. Validate and transform records through Pydantic migration models.
5. Write EnergyAgent state atomically.
6. Compare record counts and stable business identifiers.
7. Start EnergyAgent in read-only/dry-run mode and run health plus parity probes.
8. Enable writes only after probes succeed.

Source files remain untouched. Legacy state is retained until the compatibility period ends. Milvus collection names and Redis key prefixes remain unchanged during the first release unless a collision is proven; later renaming uses dual-read and verified copy rather than destructive recreation.

## Request and Event Data Flows

### Alert ingest

```text
external alert or Prometheus
  -> FastAPI request/collector
  -> validation and source normalization
  -> idempotency lookup
  -> alert merge
  -> routing and silence evaluation
  -> transactional/atomic repository write
  -> timeline and observability event
  -> response
```

### Automatic remediation

```text
authorized request
  -> load alert and fault context
  -> retrieve solution evidence through existing RAG
  -> validate evidence and solution mapping
  -> construct allowlisted command
  -> idempotency check
  -> dry-run audit or live gateway send
  -> append alert timeline event
  -> response
```

### Chat

```text
request
  -> Redis recent/session state
  -> optional Mem0 search
  -> current RAG retrieval and tools
  -> LangGraph response
  -> Redis checkpoint
  -> best-effort Mem0 write
  -> response/SSE
```

## Error Handling and Safety

- External Prometheus, Milvus, Redis, MCP, Mem0, model, MySQL, and cloud-command calls have explicit timeouts.
- Provider failures become typed domain/application errors and structured observability events.
- Secret values, passwords, tokens, DSNs, raw authorization headers, and full command payload credentials are redacted at the logging boundary.
- Atomic local writes preserve the previous valid file on validation or write failure.
- Ingest and command operations use idempotency keys.
- Partial upload and indexing success remains explicit.
- Health reports component status without disclosing connection strings or credentials.
- Live remediation cannot be enabled by importing a source file alone; it requires the explicit runtime mode plus authorization configuration.

## Frontend Design

The current static frontend remains the delivery model. It gains views for:

- alert workbench and timeline;
- ownership actions and notes;
- silence rules;
- escalation, aggregation, schedules, and routing;
- single-alert analysis and remediation status;
- RAG evaluation datasets and runs;
- component health.

Frontend modules consume documented API client functions rather than embedding fetch logic throughout one large class. The UI displays `EnergyAgent`; business terms such as “on-call”, “值班”, and routing remain. The old browser storage key is migrated on first load.

## Observability and Operations

EnergyAgent adopts the useful intent of the source observability stack without copying stale GoFrame deployment templates:

- OpenTelemetry traces and metrics;
- structured application and audit logs;
- Loki log storage;
- Tempo trace storage;
- Grafana datasources and EnergyAgent dashboards;
- dual legacy/current service selectors during compatibility;
- an application Dockerfile;
- one local compose topology containing the application, Redis, etcd, MinIO, Milvus, Attu, and optional observability profile;
- Python alert mock/check tools covering ingest through remediation.

The structured log investigation CLI supports trace, span, request, session, conversation, alert, fault, and command correlation without printing secrets.

## Testing Strategy

### Baseline gate

Before each package, run and record the current Python test suite. A package may not knowingly regress existing chat, RAG, Redis, MCP, AIOps, upload, health, or stack behavior.

### Contract tests

FastAPI tests cover every migrated route with success, validation, authorization, not-found, conflict/idempotency, dependency failure, and persistence side effects. SSE tests assert event ordering and terminal events.

### Domain tests

Port the behavior represented by the source Go tests into Python tests for:

- alert merge, touched/resolved behavior, events, ownership, and notes;
- silence matching and lifecycle;
- password policy, authentication, role authorization, and session expiry;
- schedules, rotations, restrictions, current-on-call, escalation, aggregation, and routing;
- solution resolution, command mapping, idempotency, dry-run/live safety, and audit;
- RAG evaluation scoring and persistence.

### Migration and security tests

- State migration fixtures verify counts, IDs, timestamps, and rollback copies.
- Configuration tests use fake values and compare fingerprints only.
- A repository scan fails if recognized real-secret patterns enter tracked files.
- File-mode and Git-ignore tests cover `.env`, local YAML, migration vaults, and backups.

### End-to-end tests

Python alert mock/check scenarios validate:

```text
mock fault
  -> alert ingest
  -> routing/ownership
  -> RAG evidence
  -> dry-run remediation
  -> event/audit verification
```

Full acceptance also verifies the observability correlation path and frontend smoke flows.

## Acceptance Criteria

The migration is complete when:

1. The product, API documentation, startup output, UI, package metadata, scripts, and service labels use EnergyAgent.
2. Current Python tests and the migrated source behavior tests pass.
3. All 48 source routes have a parity-ledger disposition and all applicable routes pass contract tests.
4. The current eight Python routes remain compatible.
5. Alert, silence, user, on-call, routing, and evaluation state survives restart and verified migration.
6. Configuration migration preserves every in-scope source and target key, resolves mappings, and drops no unknown key silently.
7. No secret value appears in tracked files, command output, reports, logs, tests, or API responses.
8. The target `.env`, local YAML, migration vault, and backups are ignored, untracked, and mode `0600` where applicable.
9. Default remediation is dry-run; live remediation requires explicit configuration and authorization.
10. Local stack health includes FastAPI, Redis, Milvus dependencies, MCP services, and optional observability components.
11. Alert mock/check and frontend smoke scenarios pass.
12. Rollback to the protected pre-migration target and source state has been rehearsed.

## Rollback

Each delivery package is merged separately and can be disabled through feature/configuration boundaries. State and configuration migrations are copy-first. Before enabling a new writer, retain:

- the protected target checkpoint;
- the source commit and local-state snapshot;
- target-before configuration fingerprints and backups;
- pre-migration state files;
- a migration ledger with record counts and checksums.

Rollback disables new writers, restores the previous configuration atomically, points services back to the prior state paths or repository adapter, and restarts the protected target version. Source data is never modified as part of rollback.

## Execution and Review Model

After this design is approved in writing, create one detailed implementation plan per delivery package. Use test-driven slices and small commits. At most three implementation agents work concurrently:

- one domain package per agent;
- no two agents modify the same files;
- the primary agent owns shared contracts, integration, parity ledger, secret-safe validation, and complete-suite verification;
- no implementation agents start until the current RAG working tree is protected.

Every package ends with contract review, code-quality review, targeted tests, full regression tests, and an updated parity ledger before the next dependent package begins.
