## 1. Logging Directory Configuration

- [x] 1.1 Add `LOG_DIR` / `Settings.log_dir` with default `runtime-logs`.
- [x] 1.2 Route Loguru file output through the configured log directory.
- [x] 1.3 Route stack-script service stdout/stderr logs through the same directory.

## 2. Verification

- [x] 2.1 Add and run focused tests for config, logger path resolution, and script log path handling.
- [x] 2.2 Run lint or syntax checks for touched files.
- [x] 2.3 Restart the project and verify logs are created under the configured directory.
