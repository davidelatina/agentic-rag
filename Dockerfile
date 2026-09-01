FROM python:3.12-slim-bookworm

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app
COPY pyproject.toml uv.lock ./
COPY src ./src
COPY prompts ./prompts
COPY data ./data

RUN uv sync --frozen --no-dev

CMD ["uv", "run", "app"]
