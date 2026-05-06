# Build stage: install Python dependencies with uv
FROM python:3.12-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

COPY pyproject.toml uv.lock ./
COPY src/ ./src/
COPY README.md ./

RUN uv sync --frozen --no-dev

# Runtime stage: minimal image with only what is needed to run
FROM python:3.12-slim AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv
COPY src/ ./src/
COPY app/ ./app/
COPY entrypoint.sh ./entrypoint.sh

RUN chmod +x entrypoint.sh \
    && mkdir -p data/raw data/processed outputs/models outputs/reports

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1

EXPOSE 8501

ENTRYPOINT ["./entrypoint.sh"]
