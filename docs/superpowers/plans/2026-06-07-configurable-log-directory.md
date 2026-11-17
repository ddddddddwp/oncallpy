---
archived-with: 2026-06-07-configure-log-directory
status: final
---
# Configurable Log Directory Plan

1. Add failing tests for settings, app logger path resolution, and stack script log handling.
2. Implement `Settings.log_dir` with default `runtime-logs`.
3. Route Loguru file logs through the configured directory.
4. Route stack script service logs through `--log-dir` / `LOG_DIR` / default `runtime-logs`.
5. Run focused tests and lint.
6. Restart the stack and verify logs are emitted under the configured directory.
