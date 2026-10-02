# Serves the committed site build (site/dist) plus the /api routes; see jev_tracker/server.py.
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy PORT=8000

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY jev_tracker ./jev_tracker
COPY crawler ./crawler
COPY site/dist ./site/dist
RUN uv sync --frozen --no-dev

EXPOSE 8000
CMD ["sh", "-c", "uv run --no-sync python -m jev_tracker.server --port $PORT"]
