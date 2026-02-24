#!/bin/bash
# Resilient full-text extraction pipeline runner.
# Auto-restarts on failure. Safe to leave unattended.
#
# Usage:
#   nohup bash scripts/run_pipeline.sh > data/raw/runner.log 2>&1 &
#   # Check progress:
#   tail -f data/raw/pipeline.log

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

DOCKER_PATH="/c/Program Files/Docker/Docker/resources/bin/docker.exe"
MAX_RETRIES=10
RETRY_DELAY=60

echo "[$(date)] Pipeline runner started"

# ── Phase 1: Download PDFs with auto-restart ──────────────────────────
echo "[$(date)] Starting Phase 1: PDF Downloads"
for attempt in $(seq 1 $MAX_RETRIES); do
    echo "[$(date)] Download attempt $attempt/$MAX_RETRIES"
    python scripts/fulltext_pipeline.py --skip-grobid --workers 3 --delay 0.3 && break
    echo "[$(date)] Download crashed, restarting in ${RETRY_DELAY}s..."
    sleep $RETRY_DELAY
done

# ── Ensure GROBID is running ─────────────────────────────────────────
echo "[$(date)] Starting Phase 2: GROBID extraction"
echo "[$(date)] Pulling GROBID Docker image (first time ~2.5GB)..."
"$DOCKER_PATH" pull lfoppiano/grobid:0.8.1 || true

# Start GROBID container
"$DOCKER_PATH" rm -f grobid 2>/dev/null || true
"$DOCKER_PATH" run -d --name grobid -p 8070:8070 --memory 4g lfoppiano/grobid:0.8.1

# Wait for GROBID to be ready
echo "[$(date)] Waiting for GROBID to start..."
for i in $(seq 1 60); do
    if curl -s http://localhost:8070/api/isalive > /dev/null 2>&1; then
        echo "[$(date)] GROBID ready after ${i}x5 seconds"
        break
    fi
    sleep 5
done

# ── Phase 2: GROBID extraction with auto-restart ─────────────────────
for attempt in $(seq 1 $MAX_RETRIES); do
    echo "[$(date)] GROBID extraction attempt $attempt/$MAX_RETRIES"

    # Make sure GROBID container is still running
    "$DOCKER_PATH" start grobid 2>/dev/null || true
    sleep 10

    python scripts/fulltext_pipeline.py --skip-download --workers 1 && break
    echo "[$(date)] GROBID extraction crashed, restarting in ${RETRY_DELAY}s..."
    sleep $RETRY_DELAY
done

# ── Stop GROBID ──────────────────────────────────────────────────────
echo "[$(date)] Stopping GROBID container"
"$DOCKER_PATH" stop grobid 2>/dev/null || true

echo "[$(date)] Pipeline runner finished!"
echo "[$(date)] Next steps: run re-chunking, re-embedding, and re-enrichment manually"
