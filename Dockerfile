# syntax=docker/dockerfile:1.7
#
# Multi-stage, non-root, production-grade image for FastAPI service
#

############################
# Builder
############################
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install build tools only in builder
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Use a virtualenv for clean runtime copy
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt ./
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

############################
# Runtime
############################
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    # Sensible defaults; can be overridden at runtime
    ENV=production \
    LOG_LEVEL=INFO \
    DATA_DIR=/data \
    DEFAULT_METRIC=cosine \
    DEFAULT_INDEX=linear \
    WEB_CONCURRENCY=1

WORKDIR /app

# Install minimal runtime deps useful for healthcheck/debug
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN addgroup --system app && adduser --system --ingroup app app

# Copy the prebuilt virtualenv
COPY --from=builder /opt/venv /opt/venv

# Copy application code
COPY app ./app

# Data directory (mounted via volume in compose)
RUN mkdir -p /data && chown -R app:app /data

USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/health || exit 1

CMD ["sh", "-c", \
    "exec gunicorn -k uvicorn.workers.UvicornWorker -w ${WEB_CONCURRENCY:-2} -b 0.0.0.0:8000 --access-logfile - --error-logfile - --timeout 60 app.main:app"]
