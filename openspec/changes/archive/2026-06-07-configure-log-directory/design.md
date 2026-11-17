## Context

The backend uses Loguru for application logs and a local Python stack script to supervise MCP and FastAPI processes. These two logging surfaces currently use different destinations. The app writes rotated logs under `logs/`, while the script writes process output files at the project root.

The desired behavior is a single operator-facing directory for local runtime logs, with a practical default and a way to override it per environment.

## Goals

- Use one configurable directory for local project logs.
- Make the default directory explicit: `runtime-logs/`.
- Preserve existing log file names where possible to avoid losing operator familiarity.
- Ensure nested log directories are created before processes start.
- Avoid restarting healthy dependency containers as part of logging changes.

## Non-Goals

- Do not change log format semantics beyond the destination path.
- Do not introduce a new logging backend, collector, or external service.
- Do not change Docker Compose data volumes or vector database storage.

## Decisions

### Decision 1: Add `LOG_DIR`

`Settings` will expose `log_dir` with default `runtime-logs`. This keeps configuration consistent with the existing Pydantic Settings model and allows `.env` or shell environments to override the destination.

### Decision 2: Keep service log file names

The stack script will write:

- `runtime-logs/server.log`
- `runtime-logs/mcp_cls.log`
- `runtime-logs/mcp_monitor.log`

The application logger will write:

- `runtime-logs/app_{time:YYYY-MM-DD}.log`

### Decision 3: Script accepts an explicit override

The stack script will accept `--log-dir`. If omitted, it will use `LOG_DIR` and then fall back to `runtime-logs`.

## Risks

- Existing tooling that tails root-level `server.log` will need to tail `runtime-logs/server.log` instead.
- If an absolute `LOG_DIR` is configured, logs will go outside the project root by design.
