---
archived-with: 2026-06-07-configure-log-directory
status: final
status: final
---
# Configurable Log Directory Design

## Summary

Use `LOG_DIR` to place all local project runtime logs in one directory. The default is `runtime-logs/` so operators can inspect FastAPI, MCP, and application logs without scanning the repository root or the old `logs/` directory.

## Design

- Add `Settings.log_dir` with default `runtime-logs`.
- Add logger path helpers in `app/utils/logger.py`.
- Create the configured directory before registering the Loguru file sink.
- Add stack script support for `--log-dir`, falling back to `LOG_DIR`, then `runtime-logs`.
- Keep service file names stable: `server.log`, `mcp_cls.log`, and `mcp_monitor.log`.

## Validation

- Unit tests cover `LOG_DIR` parsing and path resolution.
- Unit tests cover stack script log path assignment and directory creation.
- Restart verification confirms runtime logs are created in the configured directory.
