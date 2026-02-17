# ── Stage 1: Build frontend ─────────────────────────────────────────
FROM node:22-slim AS frontend

WORKDIR /frontend
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm ci --no-audit --no-fund 2>/dev/null || npm install --no-audit --no-fund
COPY frontend/ .
RUN npm run build

# ── Stage 2: Build Python dependencies ──────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /build

# Install build deps for native extensions (numpy, scipy, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ── Stage 3: Runtime ────────────────────────────────────────────────
FROM python:3.12-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# NLTK data (punkt tokenizer for BM25)
RUN python -m nltk.downloader -d /usr/share/nltk_data punkt punkt_tab

ENV NLTK_DATA=/usr/share/nltk_data
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Create non-root user
RUN groupadd --gid 1000 appuser && \
    useradd --uid 1000 --gid 1000 --create-home appuser

# Copy application code
COPY src/ ./src/
COPY scripts/ ./scripts/

# Copy frontend build
COPY --from=frontend /frontend/dist ./frontend/dist

# Data dir (mount or copy at runtime)
RUN mkdir -p /app/data && chown appuser:appuser /app/data
VOLUME /app/data

# Model cache (sentence-transformers downloads on first run)
ENV HF_HOME=/app/.cache/huggingface
RUN mkdir -p /app/.cache/huggingface && chown -R appuser:appuser /app/.cache

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health')" || exit 1

CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
