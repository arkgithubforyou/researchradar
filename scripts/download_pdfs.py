"""Download ACL Anthology PDFs. Resumable — safe to interrupt and restart.

Usage:
    python scripts/download_pdfs.py                     # Download all
    python scripts/download_pdfs.py --max-papers 100    # Test with 100
    python scripts/download_pdfs.py --workers 5         # More parallel downloads
"""

import argparse
import json
import logging
import sys
import time
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import get_config
from src.storage.sqlite_db import SQLiteDB

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

PDF_DIR = Path("data/raw/pdfs")
USER_AGENT = "ResearchRadar/0.6 (academic research tool)"


def download_one(paper_id: str, url: str) -> tuple[str, bool, str]:
    """Download a single PDF. Returns (paper_id, success, message)."""
    safe_name = paper_id.replace("::", "_").replace("/", "_")
    pdf_path = PDF_DIR / f"{safe_name}.pdf"

    if pdf_path.exists() and pdf_path.stat().st_size > 1000:
        return (paper_id, True, "cached")

    try:
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        resp = urllib.request.urlopen(req, timeout=30)
        data = resp.read()
        if len(data) < 1000:
            return (paper_id, False, f"too small ({len(data)} bytes)")
        pdf_path.write_bytes(data)
        return (paper_id, True, "ok")
    except Exception as e:
        return (paper_id, False, str(e)[:100])


def main():
    parser = argparse.ArgumentParser(description="Download ACL Anthology PDFs")
    parser.add_argument("--max-papers", type=int, default=None)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--delay", type=float, default=0.3)
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

    PDF_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Papers to download: %d (workers=%d)", len(rows), args.workers)

    downloaded = 0
    cached = 0
    failed = 0
    failed_ids = []

    for batch_start in range(0, len(rows), 100):
        batch = rows[batch_start:batch_start + 100]

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = []
            for paper_id, url in batch:
                time.sleep(args.delay / args.workers)
                futures.append(executor.submit(download_one, paper_id, url))

            for future in as_completed(futures):
                pid, success, msg = future.result()
                if success:
                    if msg == "cached":
                        cached += 1
                    else:
                        downloaded += 1
                else:
                    failed += 1
                    failed_ids.append(pid)
                    if failed <= 20:
                        logger.warning("Failed: %s — %s", pid, msg)

        total = downloaded + cached + failed
        if total % 1000 < 100 or batch_start + 100 >= len(rows):
            logger.info(
                "Progress: %d/%d (new=%d, cached=%d, failed=%d)",
                total, len(rows), downloaded, cached, failed,
            )

    logger.info("=== Done: %d downloaded, %d cached, %d failed ===",
                downloaded, cached, failed)

    if failed_ids:
        fail_path = PDF_DIR / "failed_downloads.json"
        fail_path.write_text(json.dumps(failed_ids, indent=2), encoding="utf-8")
        logger.info("Failed IDs saved to %s", fail_path)


if __name__ == "__main__":
    main()
