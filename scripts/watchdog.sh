#!/bin/bash
# Watchdog for PyMuPDF-based full-text extraction pipeline.
# Auto-restarts on crash. Safe to leave unattended overnight.
#
# Usage:
#   nohup bash scripts/watchdog.sh > data/raw/watchdog.log 2>&1 &
#   # Check progress:
#   tail -f data/raw/extract_fulltext.log

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

MAX_RETRIES=20
RETRY_DELAY=60

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

log "Watchdog started"
log "Working directory: $PROJECT_DIR"

# ── Wait for any existing extract_fulltext.py process ───────────────
EXISTING_PID=$(tasklist 2>/dev/null | grep -i python | awk '{print $2}')
if [ -n "$EXISTING_PID" ]; then
    # Check if it's running extract_fulltext
    CMDLINE=$(wmic process where "processid=$EXISTING_PID" get commandline 2>/dev/null | grep -i extract_fulltext || true)
    if [ -n "$CMDLINE" ]; then
        log "Found existing extract_fulltext.py (PID=$EXISTING_PID), waiting for it..."
        while tasklist 2>/dev/null | grep -q "$EXISTING_PID"; do
            sleep 30
        done
        log "Existing process finished, checking if restart needed..."
    fi
fi

for attempt in $(seq 1 $MAX_RETRIES); do
    log "=== Attempt $attempt/$MAX_RETRIES ==="

    # Check if pipeline is complete (both phases done)
    DONE=$(python -c "
import json, sys
try:
    d = json.load(open('data/raw/fulltext_progress.json'))
    dl = len(d.get('downloaded', []))
    ex = len(d.get('extracted', []))
    fe = len(d.get('failed_extract', []))
    print(f'dl={dl} ex={ex} fe={fe}')
    # If extraction phase is done (extracted + failed_extract == downloaded)
    if ex + fe >= dl and dl > 0:
        sys.exit(0)
    else:
        sys.exit(1)
except Exception as e:
    print(f'Error: {e}')
    sys.exit(1)
" 2>&1)
    if [ $? -eq 0 ]; then
        log "Pipeline already complete! ($DONE)"
        break
    fi
    log "Progress: $DONE"

    log "Starting extract_fulltext.py (workers=3, delay=0.3)"
    python scripts/extract_fulltext.py --workers 3 --delay 0.3
    EXIT_CODE=$?

    if [ $EXIT_CODE -eq 0 ]; then
        log "Pipeline completed successfully!"
        break
    else
        log "Pipeline crashed with exit code $EXIT_CODE"
        if [ $attempt -lt $MAX_RETRIES ]; then
            log "Restarting in ${RETRY_DELAY}s..."
            sleep $RETRY_DELAY
        fi
    fi
done

log "Watchdog finished after $attempt attempts"

# Print summary
python -c "
import json
d = json.load(open('data/raw/fulltext_progress.json'))
print(f'Downloaded: {len(d.get(\"downloaded\",[]))}')
print(f'Extracted: {len(d.get(\"extracted\",[]))}')
print(f'Failed DL: {len(d.get(\"failed_download\",[]))}')
print(f'Failed Extract: {len(d.get(\"failed_extract\",[]))}')
"

# ── Reboot to enable WSL2/Docker for GROBID ──────────────────────────
log "Pipeline done. Rebooting in 60s to enable WSL2 for Docker/GROBID..."
shutdown.exe /r /t 60
