# Redis Checkpoint Design

## Goal

Persist LangGraph checkpoints in Redis instead of process memory. The app should keep using `session_id` as LangGraph `thread_id`, but checkpoints must survive FastAPI process restarts when Redis is running.

## Context

The project currently uses `MemorySaver` in three LangGraph services:

- `RAGAgentService`
- `AIOpsService`
- `MultiAgentAIOpsService`

Those checkpoints are isolated by `thread_id`, which is already derived from request `session_id`. Documentation also states that the current backend session state is memory-only and is lost after restart.

## Chosen Approach

Use a centralized Redis checkpointer factory.

Add `REDIS_URL` support through `app/config.py`, defaulting to:

```text
redis://:your-redis-password@localhost:6379/0
```

Add a new core module responsible for creating LangGraph Redis checkpointers. All LangGraph services should import this factory instead of constructing `MemorySaver` directly.

## Behavior

When the app starts, services compile their LangGraph workflows with a Redis-backed checkpointer.

If Redis is unreachable or the Redis checkpoint dependency is missing, startup should fail loudly with an actionable error. It should not silently fall back to `MemorySaver`, because that would make checkpoint persistence look enabled while data is only stored in memory.

The existing API contract remains unchanged:

- request bodies continue to accept `session_id`
- LangGraph config continues to use `{"configurable": {"thread_id": session_id}}`
- chat session history APIs continue to read and clear checkpoints through the service checkpointer
- AIOps streaming event shape does not change

## Components

### Configuration

`app/config.py` gets:

- `redis_url: str`

The value is loaded from `REDIS_URL`, with the local WSL2 Docker default above.

### Checkpoint Factory

New module:

- `app/core/checkpoint.py`

Responsibilities:

- create a Redis checkpointer from `config.redis_url`
- hide version-specific setup details from services
- raise clear runtime errors when Redis checkpoint support is not installed or Redis cannot be initialized

### Service Integration

Replace direct `MemorySaver()` usage in:

- `app/services/rag_agent_service.py`
- `app/services/aiops_service.py`
- `app/services/multi_agent_aiops_service.py`

Each service should call the centralized factory during initialization and compile its graph with the returned checkpointer.

`MultiAgentAIOpsService._ensure_graph()` should recreate the checkpointer through the same factory if the instance was created without normal initialization in tests or after a lazy retry path.

## Dependencies

If the current installed LangGraph stack does not include Redis checkpoint support, add the official Redis checkpoint dependency to `pyproject.toml`.

The implementation should prefer the current official LangGraph Redis checkpointer API available in the environment, rather than hand-writing Redis serialization.

## Tests

Add focused tests that do not require a live Redis server:

- config exposes the default Redis URL and honors `REDIS_URL`
- services use the centralized checkpointer factory instead of `MemorySaver`
- graph compilation receives the created checkpointer
- session history and clear-session code continue to call the checkpointer through the same public methods where applicable

Do not add integration tests that require Docker Redis unless the project already has that pattern.

## Documentation

Update local documentation to state:

- backend checkpoints are Redis-backed
- default local Redis URL is `redis://:your-redis-password@localhost:6379/0`
- Redis must be running in local WSL2 Docker before starting the app
- service restarts preserve checkpoint data while Redis data remains available

## Out of Scope

- New checkpoint inspection APIs
- Checkpoint migration from existing `MemorySaver`
- Frontend changes
- Production Redis deployment topology
- Automatic Redis container orchestration

## Risks

LangGraph Redis checkpoint package APIs can differ across versions. The implementation should isolate this in `app/core/checkpoint.py` so future package changes do not require touching all services.

Redis availability becomes a startup/runtime dependency. This is intentional for correctness because persistence is a required behavior.
