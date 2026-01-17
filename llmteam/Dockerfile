# LLMTeam Docker Image
# Enterprise AI Workflow Runtime
#
# Build:
#   docker build -t llmteam:latest .
#
# Run:
#   docker run -p 8000:8000 llmteam:latest
#
# Run with environment:
#   docker run -p 8000:8000 \
#     -e LLMTEAM_LICENSE_KEY=xxx \
#     -e DATABASE_URL=postgresql://... \
#     llmteam:latest

FROM python:3.11-slim AS builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml README.md ./
COPY src/ src/

# Build wheel
RUN pip install --no-cache-dir build && \
    python -m build --wheel

# -----------------------------------------------------------------------------
# Production image
# -----------------------------------------------------------------------------
FROM python:3.11-slim

LABEL org.opencontainers.image.title="LLMTeam"
LABEL org.opencontainers.image.description="Enterprise AI Workflow Runtime"
LABEL org.opencontainers.image.version="2.0.0"
LABEL org.opencontainers.image.vendor="LLMTeam"
LABEL org.opencontainers.image.source="https://github.com/KirilinVS/llmteam"

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy wheel from builder
COPY --from=builder /build/dist/*.whl /tmp/

# Install llmteam with all optional dependencies
RUN pip install --no-cache-dir /tmp/*.whl[all] && \
    rm /tmp/*.whl

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash llmteam && \
    chown -R llmteam:llmteam /app

USER llmteam

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV LLMTEAM_LOG_FORMAT=json
ENV LLMTEAM_HOST=0.0.0.0
ENV LLMTEAM_PORT=8000

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command: run API server
CMD ["llmteam", "serve", "--host", "0.0.0.0", "--port", "8000"]
