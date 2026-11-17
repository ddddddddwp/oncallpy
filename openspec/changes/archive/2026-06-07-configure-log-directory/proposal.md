## Why

Project logs are currently split between `logs/app_*.log` and root-level service files such as `server.log`, `mcp_cls.log`, and `mcp_monitor.log`. This makes local troubleshooting inconsistent and leaves runtime artifacts mixed into the repository root.

## What Changes

- Add a configurable application log directory with `LOG_DIR`.
- Default runtime logs to `runtime-logs/` under the project root.
- Route Loguru application logs into the configured directory.
- Route local stack script stdout/stderr logs for FastAPI and MCP services into the same directory.
- Keep dependency services such as Docker, Milvus, Redis, Attu, and MinIO managed as before.

## Impact

- Affected configuration:
  - `app/config.py`
  - optional `LOG_DIR` environment variable
- Affected runtime logging:
  - `app/utils/logger.py`
  - `scripts/oncallpy_stack.py`
- Affected tests:
  - focused config/logger tests
  - stack restart script tests
