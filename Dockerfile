# Multi-stage build. For multi-arch (Apple Silicon dev + amd64 servers):
#   docker buildx build --platform linux/amd64,linux/arm64 -t rag-insurance-api .
FROM python:3.11-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /usr/local/bin/uv
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-install-project --no-dev
COPY src ./src
RUN uv sync --frozen --no-dev

FROM python:3.11-slim
WORKDIR /app
COPY --from=builder /app/.venv /app/.venv
COPY src ./src
# HF_HOME points at a volume (see docker-compose.yml) so model weights are
# downloaded once and never baked into the image.
ENV PATH="/app/.venv/bin:$PATH" \
    HF_HOME=/data/hf \
    PYTHONUNBUFFERED=1
EXPOSE 8000
CMD ["uvicorn", "rag_insurance.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
