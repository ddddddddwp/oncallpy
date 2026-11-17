---
name: manage-oncallpy-stack
description: Use when starting, checking, restarting, or recovering the local oncallpy project stack in WSL2/Docker, especially FastAPI, the static frontend, MCP services, Milvus, Attu, MinIO, or Redis.
---

# Manage Oncallpy Stack

## Overview

Use the project script instead of hand-running Makefile targets when the user asks to start,
restart, or recover this project locally. The frontend is served by FastAPI from `static/`, so
there is no separate frontend process.

## Command

Run from the repository root:

```bash
scripts/oncallpy_stack.py restart
```

This checks Docker, starts missing dependency components, then stops and restarts only the
application services:

- `mcp_servers/monitor_server.py` on `127.0.0.1:8004`
- `mcp_servers/cls_server.py` on `127.0.0.1:8003`
- `uvicorn app.main:app` on `127.0.0.1:9900`

Dependency behavior:

- Milvus stack uses `docker compose -f vector-database.yml up -d` only if the Milvus containers
  are missing, stopped, or unhealthy.
- Redis is checked through the configured Redis URL, defaulting to `localhost:6379`; if closed,
  the script starts an existing Docker container whose name matches `redis`.
- Dependencies are not restarted when already healthy/running.

To start dependencies without restarting application services:

```bash
scripts/oncallpy_stack.py deps
```

## Verification

After restart, verify:

```bash
curl -s http://127.0.0.1:9900/health
curl -s -o /tmp/oncallpy-root.html -w '%{http_code}\n' http://127.0.0.1:9900/
docker ps --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}'
```

Expected local URLs:

- Web UI: `http://localhost:9900`
- API docs: `http://localhost:9900/docs`
- Attu Milvus UI: `http://localhost:8000`
- MinIO console: `http://localhost:9001` (`admin` / `minioadmin`)

## Notes

Avoid plain `nohup ... &` from Codex tool sessions for these services; those jobs may be cleaned
up when the tool command exits. The script starts child processes with `start_new_session=True`
and writes the existing PID files (`server.pid`, `mcp_cls.pid`, `mcp_monitor.pid`).
