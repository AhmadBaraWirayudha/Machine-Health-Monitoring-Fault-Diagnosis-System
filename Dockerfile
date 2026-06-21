# =============================================================
# Dockerfile — CBM Health Monitoring System
# =============================================================
# Multi-stage build: keeps the final image lean.
#
# Build:
#   docker build -t cbm-system .
#
# Run full pipeline:
#   docker run --rm -v $(pwd)/data:/app/data cbm-system pipeline
#
# Run API:
#   docker run -p 8000:8000 -v $(pwd)/data:/app/data cbm-system api
#
# Run dashboard:
#   docker run -p 8501:8501 -v $(pwd)/data:/app/data cbm-system dashboard

# ── Stage 1: builder ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# System deps for scipy / matplotlib
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libgomp1 && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements-extra.txt ./
RUN pip install --upgrade pip && \
    pip wheel --no-cache-dir --wheel-dir /wheels -r requirements.txt && \
    pip wheel --no-cache-dir --wheel-dir /wheels -r requirements-extra.txt


# ── Stage 2: runtime ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

LABEL maintainer="CBM Engineering Portfolio"
LABEL description="Machine Health Monitoring & Fault Diagnosis System"
LABEL version="1.0.0"

WORKDIR /app

# Runtime system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 curl && \
    rm -rf /var/lib/apt/lists/*

# Copy pre-built wheels from builder
COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir --no-index --find-links=/wheels /wheels/* && \
    rm -rf /wheels

# Copy project source
COPY . .

# Create data directories (volume mount points)
RUN mkdir -p data/raw data/processed models reports/plots logs

# Non-root user for security
RUN useradd -m -u 1000 cbmuser && chown -R cbmuser:cbmuser /app
USER cbmuser

# Expose ports
EXPOSE 8000   
EXPOSE 8501   

# Health check for API
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Entrypoint dispatcher
COPY docker-entrypoint.sh /usr/local/bin/entrypoint.sh

ENTRYPOINT ["python", "-c", "\
import sys, os; \
cmd = sys.argv[1] if len(sys.argv) > 1 else 'pipeline'; \
cmds = { \
  'pipeline':  'import subprocess; subprocess.run([\"python\", \"main.py\"])', \
  'api':       'import subprocess; subprocess.run([\"uvicorn\", \"src.api.main:app\", \"--host\", \"0.0.0.0\", \"--port\", \"8000\"])', \
  'dashboard': 'import subprocess; subprocess.run([\"streamlit\", \"run\", \"src/dashboard/app.py\", \"--server.port\", \"8501\", \"--server.address\", \"0.0.0.0\"])', \
  'test':      'import subprocess; subprocess.run([\"pytest\", \"tests/\", \"-v\"])', \
}; \
exec(cmds.get(cmd, cmds[\"pipeline\"]))"]

CMD ["pipeline"]
