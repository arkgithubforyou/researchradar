"""Watchdog for GROBID extraction pipeline.

Monitors and auto-restarts the GROBID pipeline if it crashes.
Also ensures the GROBID Docker container stays running.

Usage:
    pythonw scripts/grobid_watchdog.py   (silent background)
    python  scripts/grobid_watchdog.py   (with console output)
"""

import json
import logging
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent
os.chdir(PROJECT_DIR)

LOG_FILE = PROJECT_DIR / "data" / "raw" / "grobid_watchdog.log"
PROGRESS_FILE = PROJECT_DIR / "data" / "raw" / "pipeline_progress.json"
DOCKER_PATH = r"C:\Program Files\Docker\Docker\resources\bin\docker.exe"
GROBID_URL = "http://localhost:8070"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(str(LOG_FILE), encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

MAX_RETRIES = 20
RETRY_DELAY = 60


def is_grobid_alive():
    try:
        r = urllib.request.urlopen(f"{GROBID_URL}/api/isalive", timeout=5)
        return r.status == 200
    except Exception:
        return False


def ensure_grobid():
    if is_grobid_alive():
        return True
    log.info("GROBID not responding, restarting container...")
    subprocess.run([DOCKER_PATH, "start", "grobid"], capture_output=True)
    for i in range(60):
        if is_grobid_alive():
            log.info("GROBID recovered after %ds", i * 5)
            return True
        time.sleep(5)
    log.error("GROBID failed to recover")
    return False


def is_pipeline_done():
    try:
        data = json.loads(PROGRESS_FILE.read_text(encoding="utf-8"))
        done = len(data.get("grobid_done", []))
        failed = len(data.get("grobid_failed", []))
        total = 26543  # known total
        return (done + failed) >= total * 0.99  # 99% threshold
    except Exception:
        return False


def main():
    log.info("GROBID watchdog started")

    for attempt in range(1, MAX_RETRIES + 1):
        if is_pipeline_done():
            log.info("Pipeline already complete!")
            break

        log.info("=== Attempt %d/%d ===", attempt, MAX_RETRIES)

        if not ensure_grobid():
            log.error("Cannot start GROBID, retrying in %ds...", RETRY_DELAY)
            time.sleep(RETRY_DELAY)
            continue

        log.info("Starting fulltext_pipeline.py --skip-download")
        proc = subprocess.Popen(
            [sys.executable, "scripts/fulltext_pipeline.py", "--skip-download", "--workers", "1"],
            cwd=str(PROJECT_DIR),
        )
        log.info("Pipeline PID=%d", proc.pid)
        proc.wait()

        if proc.returncode == 0:
            log.info("Pipeline completed successfully!")
            break
        else:
            log.warning("Pipeline exited with code %d", proc.returncode)
            if attempt < MAX_RETRIES:
                log.info("Restarting in %ds...", RETRY_DELAY)
                time.sleep(RETRY_DELAY)

    # Cleanup: stop GROBID container
    log.info("Stopping GROBID container...")
    subprocess.run([DOCKER_PATH, "stop", "grobid"], capture_output=True)
    subprocess.run([DOCKER_PATH, "rm", "grobid"], capture_output=True)

    log.info("Watchdog finished. Shutting down machine in 60s...")
    subprocess.run(["shutdown", "/s", "/t", "60"], capture_output=True)


if __name__ == "__main__":
    main()
