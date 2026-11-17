# Local Stack Dependency Recovery Design

## Goal

Make the existing local startup commands reliably restore every dependency needed by the alarmmock chat flow, without adding another startup script. The repaired stack must start Mem0 on port 8888, OnCall MCP on port 12345, and EnergyAgent with authenticated access to the existing `1Panel-redis-VrI6` instance.

Phase 2 code cleanup starts only after this recovery work passes its integration checks.

## Scope

Modify only existing startup and test files:

- `/root/code/oncall/OnCallAgent/scripts/restart_local_stack.sh`
- `/root/code/oncall/OnCallAgent/scripts/start_mcp.sh` when needed for reliable restart behavior
- `/root/code/oncallpy/scripts/energyagent_stack.py`
- `/root/code/oncallpy/tests/test_oncallpy_stack_script.py`
- ignored runtime configuration in `/root/code/oncallpy/.env`

No new startup script will be created. Mem0 source code and its Compose definition under `/root/code/mem0/server` remain unchanged.

## Design

### OnCallAgent startup

Extend `restart_local_stack.sh` so one invocation performs these ordered actions:

1. Load `.env.local` before dependency decisions.
2. If the configured local Mem0 URL is enabled, run the existing `/root/code/mem0/server/docker-compose.api.yaml` with `docker compose up -d` and wait for PostgreSQL and the Mem0 HTTP endpoint on port 8888.
3. Stop stale OnCall MCP processes together with the other local services.
4. Start the OnCall backend on 6872.
5. Start the existing OnCall MCP server on 12345 and verify that the port is listening.
6. Start alarmcheck on 19090 and build `/tmp/alarmmock` as today.

The Mem0 Compose directory may be overridden by an environment variable, but its default is derived from the current repository layout rather than duplicated as an unrelated absolute path.

### Redis configuration and authentication

Treat the running `1Panel-redis-VrI6` container as the local Redis source of truth.

- Reconcile `REDIS_PASSWORD` and `REDIS_URL` in the ignored `/root/code/oncallpy/.env` with that container's current `--requirepass` value.
- Never print the password or include it in tracked files.
- Change `energyagent_stack.py` to resolve `REDIS_URL` in this order: command-line option, process environment, project `.env`, then the passwordless local default.
- Replace the current TCP-only readiness decision with an authenticated Redis `PING`. An open port with rejected credentials must fail dependency validation instead of reporting `redis: ready` and allowing FastAPI to fail later.
- If Redis is stopped, retain the existing behavior that starts an existing container whose name matches `redis`, then wait for an authenticated ping.

### Failure behavior

Startup exits non-zero with a component-specific message when Mem0, MCP, or authenticated Redis cannot become ready. Messages may identify the host, port, and container, but must not expose credentials.

## Verification

Before implementation, add or update tests so the old TCP-only Redis behavior fails the new authentication expectations. Then verify:

- Existing EnergyAgent stack tests, including `.env` URL precedence and authenticated Redis readiness.
- Shell syntax for both existing OnCallAgent startup scripts.
- Mem0 PostgreSQL and API containers are running; port 8888 responds.
- OnCall MCP listens on 12345.
- EnergyAgent `/health` on 9900 reports Milvus and Redis healthy.
- OnCallAgent 6872 and alarmcheck 19090 respond.
- The alarmmock INVT injection and continuous chat flow complete without 8888 or 12345 connection-refused stage errors.

## Non-goals

- No new orchestration script.
- No replacement Redis container.
- No modification of Mem0 application code.
- No broad dead-code deletion in this phase.
- No secrets committed to Git or emitted in test output.
