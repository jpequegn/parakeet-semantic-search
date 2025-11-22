# Multi-stage build for optimized Docker image
# Stage 1: Builder
FROM python:3.12-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies in a virtual environment
RUN python -m venv /opt/venv && \
    . /opt/venv/bin/activate && \
    pip install --no-cache-dir -r requirements.txt

# Stage 2: Runtime
FROM python:3.12-slim

WORKDIR /app

# Install runtime dependencies only (no build tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Copy application code
COPY src/ src/
COPY apps/ apps/
COPY scripts/ scripts/
COPY pyproject.toml setup.py README.md ./

# Install the package (before switching to non-root user)
RUN /opt/venv/bin/pip install -e .

# Create directories for data and logs
RUN mkdir -p /app/data /app/logs

# Set environment variables
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    LOG_LEVEL=info \
    CACHE_ENABLED=true \
    CACHE_TTL=3600 \
    API_HOST=0.0.0.0 \
    API_PORT=8000

# Set user for security (non-root)
RUN useradd -m -u 1000 parakeet && \
    chown -R parakeet:parakeet /app

USER parakeet

# Expose ports
EXPOSE 8000 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command (REST API)
# Override with docker run -c "streamlit run apps/streamlit_app.py" for web UI
CMD ["uvicorn", "apps.fastapi_app:app", "--host", "0.0.0.0", "--port", "8000"]
