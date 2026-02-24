"""Full-text extraction pipeline: download PDFs → GROBID → update DB.

Designed to run unattended. Logs everything to data/raw/pipeline.log.
Resumable — safe to interrupt and restart.

Usage:
    python scripts/fulltext_pipeline.py                  # Full pipeline
    python scripts/fulltext_pipeline.py --max-papers 50  # Quick test
    python scripts/fulltext_pipeline.py --skip-download   # GROBID only
    python scripts/fulltext_pipeline.py --skip-grobid     # Download only
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from xml.etree import ElementTree

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import get_config
from src.storage.sqlite_db import SQLiteDB

# ── Logging to both file and console ──────────────────────────────────
LOG_FILE = Path("data/raw/pipeline.log")
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(str(LOG_FILE), encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

PDF_DIR = Path("data/raw/pdfs")
TEI_DIR = Path("data/raw/tei")
PROGRESS_FILE = Path("data/raw/pipeline_progress.json")
USER_AGENT = "ResearchRadar/0.6 (academic research tool)"

GROBID_URL = "http://localhost:8070"
DOCKER_PATH = r"C:\Program Files\Docker\Docker\resources\bin\docker.exe"


# ── Progress tracking ─────────────────────────────────────────────────

def load_progress() -> dict:
    if PROGRESS_FILE.exists():
        return json.loads(PROGRESS_FILE.read_text(encoding="utf-8"))
    return {"downloaded": set(), "grobid_done": set(), "grobid_failed": set(), "db_updated": set()}


def save_progress(progress: dict):
    # Convert sets to lists for JSON serialization
    serializable = {k: list(v) if isinstance(v, set) else v for k, v in progress.items()}
    PROGRESS_FILE.write_text(json.dumps(serializable), encoding="utf-8")


def deserialize_progress(raw: dict) -> dict:
    return {k: set(v) if isinstance(v, (list, set)) else v for k, v in raw.items()}


# ── Phase 1: Download PDFs ───────────────────────────────────────────

def download_one(paper_id: str, url: str) -> tuple[str, bool, str]:
    safe_name = paper_id.replace("::", "_").replace("/", "_")
    pdf_path = PDF_DIR / f"{safe_name}.pdf"

    if pdf_path.exists() and pdf_path.stat().st_size > 1000:
        return (paper_id, True, "cached")

    try:
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        resp = urllib.request.urlopen(req, timeout=30)
        data = resp.read()
        if len(data) < 1000:
            return (paper_id, False, f"too small ({len(data)}B)")
        pdf_path.write_bytes(data)
        return (paper_id, True, "ok")
    except Exception as e:
        return (paper_id, False, str(e)[:80])


def download_pdfs(rows: list, progress: dict, workers: int = 3, delay: float = 0.3):
    to_download = [(pid, url) for pid, url in rows if pid not in progress["downloaded"]]
    logger.info("=== Phase 1: Download %d PDFs (workers=%d) ===", len(to_download), workers)

    new = 0
    cached = 0
    failed = 0

    for batch_start in range(0, len(to_download), 100):
        batch = to_download[batch_start:batch_start + 100]

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = []
            for pid, url in batch:
                time.sleep(delay / workers)
                futures.append(executor.submit(download_one, pid, url))

            for future in as_completed(futures):
                pid, success, msg = future.result()
                if success:
                    progress["downloaded"].add(pid)
                    if msg == "cached":
                        cached += 1
                    else:
                        new += 1
                else:
                    failed += 1
                    if failed <= 20:
                        logger.warning("Download fail: %s — %s", pid, msg)

        total = new + cached + failed
        if total % 2000 < 100 or batch_start + 100 >= len(to_download):
            logger.info("Download: %d/%d (new=%d cached=%d failed=%d)",
                       total, len(to_download), new, cached, failed)
            save_progress(progress)

    save_progress(progress)
    logger.info("=== Download done: %d new, %d cached, %d failed ===", new, cached, failed)


# ── Phase 2: GROBID extraction ───────────────────────────────────────

def ensure_grobid_running():
    """Start GROBID Docker container if not running."""
    # Check if GROBID is already responding
    try:
        resp = urllib.request.urlopen(f"{GROBID_URL}/api/isalive", timeout=5)
        if resp.status == 200:
            logger.info("GROBID already running at %s", GROBID_URL)
            return True
    except Exception:
        pass

    # Start GROBID via Docker
    logger.info("Starting GROBID Docker container...")
    try:
        subprocess.run(
            [DOCKER_PATH, "run", "-d", "--rm",
             "--name", "grobid",
             "-p", "8070:8070",
             "--memory", "4g",
             "lfoppiano/grobid:0.8.1"],
            check=True, capture_output=True, text=True,
        )
    except subprocess.CalledProcessError:
        # Maybe container already exists but stopped
        try:
            subprocess.run(
                [DOCKER_PATH, "start", "grobid"],
                check=True, capture_output=True, text=True,
            )
        except subprocess.CalledProcessError as e:
            logger.error("Failed to start GROBID: %s", e.stderr)
            return False

    # Wait for GROBID to be ready
    for i in range(60):
        try:
            resp = urllib.request.urlopen(f"{GROBID_URL}/api/isalive", timeout=5)
            if resp.status == 200:
                logger.info("GROBID ready after %ds", i * 5)
                return True
        except Exception:
            pass
        time.sleep(5)

    logger.error("GROBID failed to start within 5 minutes")
    return False


def grobid_extract_one(pdf_path: Path) -> str | None:
    """Send a PDF to GROBID and return extracted text."""
    try:
        with open(pdf_path, "rb") as f:
            pdf_data = f.read()

        import http.client
        import mimetypes

        boundary = "----GROBIDBoundary"
        body = (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="input"; filename="{pdf_path.name}"\r\n'
            f"Content-Type: application/pdf\r\n\r\n"
        ).encode("utf-8") + pdf_data + f"\r\n--{boundary}--\r\n".encode("utf-8")

        req = urllib.request.Request(
            f"{GROBID_URL}/api/processFulltextDocument",
            data=body,
            headers={
                "Content-Type": f"multipart/form-data; boundary={boundary}",
                "Accept": "application/xml",
            },
        )
        resp = urllib.request.urlopen(req, timeout=120)
        tei_xml = resp.read().decode("utf-8")

        # Parse TEI XML to extract plain text
        return parse_tei_to_text(tei_xml)

    except Exception as e:
        logger.debug("GROBID error for %s: %s", pdf_path.name, str(e)[:100])
        return None


def parse_tei_to_text(tei_xml: str) -> str | None:
    """Extract plain text from GROBID TEI-XML output."""
    try:
        ns = {"tei": "http://www.tei-c.org/ns/1.0"}
        root = ElementTree.fromstring(tei_xml)

        sections = []

        # Get abstract
        abstract = root.find(".//tei:profileDesc/tei:abstract", ns)
        if abstract is not None:
            text = " ".join(abstract.itertext()).strip()
            if text:
                sections.append(text)

        # Get body text
        body = root.find(".//tei:text/tei:body", ns)
        if body is not None:
            for div in body.findall(".//tei:div", ns):
                # Get section header
                head = div.find("tei:head", ns)
                if head is not None and head.text:
                    sections.append(f"\n{head.text.strip()}\n")

                # Get paragraphs
                for p in div.findall("tei:p", ns):
                    text = " ".join(p.itertext()).strip()
                    if text:
                        sections.append(text)

        full_text = "\n\n".join(sections)
        return full_text if len(full_text) > 200 else None

    except Exception as e:
        logger.debug("TEI parse error: %s", str(e)[:100])
        return None


def grobid_extract_all(rows: list, progress: dict):
    """Run GROBID on all downloaded PDFs."""
    to_extract = []
    for pid, _ in rows:
        if pid in progress["grobid_done"] or pid in progress["grobid_failed"]:
            continue
        safe_name = pid.replace("::", "_").replace("/", "_")
        if (PDF_DIR / f"{safe_name}.pdf").exists():
            to_extract.append(pid)
    logger.info("=== Phase 2: GROBID extract %d PDFs ===", len(to_extract))

    if not ensure_grobid_running():
        logger.error("Cannot start GROBID. Aborting extraction.")
        return

    extracted = 0
    failed = 0

    for i, pid in enumerate(to_extract):
        safe_name = pid.replace("::", "_").replace("/", "_")
        pdf_path = PDF_DIR / f"{safe_name}.pdf"

        if not pdf_path.exists():
            progress["grobid_failed"].add(pid)
            failed += 1
            continue

        full_text = grobid_extract_one(pdf_path)

        if full_text:
            # Save TEI text to file for caching
            tei_path = TEI_DIR / f"{safe_name}.txt"
            tei_path.write_text(full_text, encoding="utf-8")
            progress["grobid_done"].add(pid)
            extracted += 1
        else:
            progress["grobid_failed"].add(pid)
            failed += 1

        if (i + 1) % 500 == 0 or i + 1 == len(to_extract):
            logger.info("GROBID: %d/%d (extracted=%d failed=%d)",
                       i + 1, len(to_extract), extracted, failed)
            save_progress(progress)

    save_progress(progress)
    logger.info("=== GROBID done: %d extracted, %d failed ===", extracted, failed)


# ── Phase 3: Update database ─────────────────────────────────────────

def update_database(rows: list, progress: dict, db: SQLiteDB):
    """Write extracted full text into SQLite."""
    to_update = [
        pid for pid, _ in rows
        if pid in progress["grobid_done"]
        and pid not in progress["db_updated"]
    ]
    logger.info("=== Phase 3: Update DB with %d full texts ===", len(to_update))

    batch = []
    updated = 0

    for pid in to_update:
        safe_name = pid.replace("::", "_").replace("/", "_")
        tei_path = TEI_DIR / f"{safe_name}.txt"

        if not tei_path.exists():
            continue

        full_text = tei_path.read_text(encoding="utf-8")
        batch.append((full_text, pid))
        progress["db_updated"].add(pid)
        updated += 1

        if len(batch) >= 500:
            conn = db.get_connection()
            conn.executemany("UPDATE papers SET full_text = ? WHERE id = ?", batch)
            conn.commit()
            conn.close()
            batch = []
            save_progress(progress)
            logger.info("DB update: %d/%d", updated, len(to_update))

    if batch:
        conn = db.get_connection()
        conn.executemany("UPDATE papers SET full_text = ? WHERE id = ?", batch)
        conn.commit()
        conn.close()

    save_progress(progress)
    logger.info("=== DB update done: %d papers ===", updated)


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Full-text extraction pipeline")
    parser.add_argument("--max-papers", type=int, default=None)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--delay", type=float, default=0.3)
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-grobid", action="store_true")
    args = parser.parse_args()

    config = get_config()
    db = SQLiteDB(config.sqlite_db_path)
    conn = db.get_connection()

    rows = conn.execute("""
        SELECT id, url FROM papers
        WHERE url IS NOT NULL AND url LIKE '%.pdf'
        ORDER BY year DESC, id
    """).fetchall()
    conn.close()

    if args.max_papers:
        rows = rows[:args.max_papers]

    logger.info("Pipeline starting: %d papers", len(rows))

    PDF_DIR.mkdir(parents=True, exist_ok=True)
    TEI_DIR.mkdir(parents=True, exist_ok=True)

    raw = load_progress()
    progress = deserialize_progress(raw)

    # Phase 1: Download
    if not args.skip_download:
        download_pdfs(rows, progress, workers=args.workers, delay=args.delay)

    # Phase 2: GROBID
    if not args.skip_grobid:
        grobid_extract_all(rows, progress)

    # Phase 3: DB update
    update_database(rows, progress, db)

    # Summary
    conn = db.get_connection()
    total = conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
    with_ft = conn.execute(
        "SELECT COUNT(*) FROM papers WHERE full_text IS NOT NULL AND full_text != ''"
    ).fetchone()[0]
    avg_len = conn.execute(
        "SELECT AVG(LENGTH(full_text)) FROM papers WHERE full_text IS NOT NULL AND full_text != ''"
    ).fetchone()[0]
    conn.close()

    logger.info("=== Pipeline complete ===")
    logger.info("Papers: %d | With full text: %d (%.1f%%) | Avg length: %s chars",
                total, with_ft, 100 * with_ft / max(total, 1),
                f"{avg_len:,.0f}" if avg_len else "N/A")


if __name__ == "__main__":
    main()
