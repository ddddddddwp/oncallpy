FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

COPY app ./app
COPY static ./static
COPY config/energyagent.yaml ./config/energyagent.yaml

RUN uv sync --frozen --no-dev \
    && groupadd --system --gid 10001 energyagent \
    && useradd --system --uid 10001 --gid energyagent --home-dir /app \
        --shell /usr/sbin/nologin energyagent \
    && chown -R energyagent:energyagent /app

ENV PATH="/app/.venv/bin:$PATH" \
    UV_NO_SYNC=1

USER energyagent

EXPOSE 9900

CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "9900"]
