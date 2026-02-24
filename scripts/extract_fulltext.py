"""Download ACL Anthology PDFs and extract full text using PyMuPDF.

Resumable: tracks progress in a JSON file so it can be safely interrupted
and restarted. Downloads PDFs to data/raw/pdfs/ and stores extracted text
directly into the SQLite database.

Usage:
    python scripts/extract_fulltext.py                     # Process all papers
    python scripts/extract_fulltext.py --max-papers 100    # Test with 100
    python scripts/extract_fulltext.py --workers 4         # Parallel downloads
    python scripts/extract_fulltext.py --skip-download      # Extract only (PDFs already downloaded)
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

import fitz  # PyMuPDF

from src.config import get_config
from src.storage.sqlite_db import SQLiteDB

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

PROGRESS_FILE = Path("data/raw/fulltext_progress.json")
PDF_DIR = Path("data/raw/pdfs")
USER_AGENT = "ResearchRadar/0.6 (academic research tool; https://github.com/arkgithubforyou/researchradar)"


def load_progress() -> dict:
    """Load progress tracking file."""
    if PROGRESS_FILE.exists():
        return json.loads(PROGRESS_FILE.read_text(encoding="utf-8"))
    return {"downloaded": [], "extracted": [], "failed_download": [], "failed_extract": []}


def save_progress(progress: dict):
    """Save progress tracking file."""
    PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
    PROGRESS_FILE.write_text(json.dumps(progress, indent=2), encoding="utf-8")


def download_pdf(paper_id: str, url: str, pdf_dir: Path) -> tuple[str, bool, str]:
    """Download a single PDF. Returns (paper_id, success, error_msg)."""
    # Sanitize filename: replace :: with _
    safe_name = paper_id.replace("::", "_").replace("/", "_")
    pdf_path = pdf_dir / f"{safe_name}.pdf"

    if pdf_path.exists() and pdf_path.stat().st_size > 1000:
        return (paper_id, True, "already exists")

    try:
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        resp = urllib.request.urlopen(req, timeout=30)
        data = resp.read()

        if len(data) < 1000:
            return (paper_id, False, f"too small ({len(data)} bytes)")

        pdf_path.write_bytes(data)
        return (paper_id, True, "downloaded")
    except Exception as e:
        return (paper_id, False, str(e))


def extract_text_from_pdf(pdf_path: Path) -> str | None:
    """Extract text from a PDF using PyMuPDF. Returns full text or None."""
    try:
        doc = fitz.open(str(pdf_path))
        pages = []
        for page in doc:
            text = page.get_text("text")
            if text.strip():
                pages.append(text)
        doc.close()

        if not pages:
            return None

        full_text = "\n\n".join(pages)

        # Basic cleanup
        # Remove excessive whitespace
        lines = full_text.split("\n")
        cleaned_lines = []
        for line in lines:
            stripped = line.strip()
            if stripped:
                cleaned_lines.append(stripped)
            elif cleaned_lines and cleaned_lines[-1] != "":
                cleaned_lines.append("")

        full_text = "\n".join(cleaned_lines)

        # Must have meaningful content (> 500 chars, not just headers/metadata)
        if len(full_text) < 500:
            return None

        return full_text
    except Exception as e:
        logger.debug("Failed to extract text from %s: %s", pdf_path, e)
        return None


def main():
    parser = argparse.ArgumentParser(description="Download PDFs and extract full text")
    parser.add_argument("--max-papers", type=int, default=None)
    parser.add_argument("--workers", type=int, default=3,
                        help="Parallel download workers (default: 3, be polite)")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip PDF download, only extract text")
    parser.add_argument("--delay", type=float, default=0.5,
                        help="Delay between downloads in seconds (default: 0.5)")
    args = parser.parse_args()

    config = get_config()
    db = SQLiteDB(config.sqlite_db_path)
    conn = db.get_connection()

    # Get all papers with URLs but no full text
    rows = conn.execute("""
        SELECT id, url FROM papers
        WHERE url IS NOT NULL AND url LIKE '%.pdf'
        AND (full_text IS NULL OR full_text = '')
        ORDER BY year DESC, id
    """).fetchall()
    conn.close()

    if args.max_papers:
        rows = rows[:args.max_papers]

    logger.info("Papers to process: %d", len(rows))

    PDF_DIR.mkdir(parents=True, exist_ok=True)
    progress = load_progress()

    # ── Phase 1: Download PDFs ────────────────────────────────────────
    if not args.skip_download:
        to_download = [
            (r[0], r[1]) for r in rows
            if r[0] not in progress["downloaded"]
            and r[0] not in progress["failed_download"]
        ]
        # Also retry previously failed ones
        to_download += [
            (r[0], r[1]) for r in rows
            if r[0] in progress["failed_download"]
        ]
        # Deduplicate
        seen = set()
        unique_download = []
        for paper_id, url in to_download:
            if paper_id not in seen:
                seen.add(paper_id)
                unique_download.append((paper_id, url))
        to_download = unique_download

        logger.info("=== Phase 1: Downloading %d PDFs (workers=%d, delay=%.1fs) ===",
                     len(to_download), args.workers, args.delay)

        downloaded = 0
        failed = 0
        skipped = 0

        # Use thread pool for parallel downloads with rate limiting
        batch_size = args.workers * 10
        for batch_start in range(0, len(to_download), batch_size):
            batch = to_download[batch_start:batch_start + batch_size]

            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                futures = {}
                for paper_id, url in batch:
                    time.sleep(args.delay / args.workers)  # distribute delay across workers
                    future = executor.submit(download_pdf, paper_id, url, PDF_DIR)
                    futures[future] = paper_id

                for future in as_completed(futures):
                    paper_id, success, msg = future.result()
                    if success:
                        if paper_id not in progress["downloaded"]:
                            progress["downloaded"].append(paper_id)
                        if msg == "already exists":
                            skipped += 1
                        else:
                            downloaded += 1
                    else:
                        if paper_id not in progress["failed_download"]:
                            progress["failed_download"].append(paper_id)
                        failed += 1
                        if failed <= 10:
                            logger.warning("Download failed: %s — %s", paper_id, msg)

            # Save progress periodically
            save_progress(progress)
            total_done = downloaded + skipped + failed
            if total_done % 500 == 0 or batch_start + batch_size >= len(to_download):
                logger.info(
                    "Download progress: %d/%d (downloaded=%d, cached=%d, failed=%d)",
                    total_done, len(to_download), downloaded, skipped, failed,
                )

        save_progress(progress)
        logger.info("=== Download complete: %d new, %d cached, %d failed ===",
                     downloaded, skipped, failed)

    # ── Phase 2: Extract text from PDFs ───────────────────────────────
    to_extract = [
        r[0] for r in rows
        if r[0] not in progress["extracted"]
        and r[0] not in progress["failed_extract"]
    ]

    logger.info("=== Phase 2: Extracting text from %d PDFs ===", len(to_extract))

    extracted = 0
    failed_extract = 0
    batch_updates = []

    for i, paper_id in enumerate(to_extract):
        safe_name = paper_id.replace("::", "_").replace("/", "_")
        pdf_path = PDF_DIR / f"{safe_name}.pdf"

        if not pdf_path.exists():
            progress["failed_extract"].append(paper_id)
            failed_extract += 1
            continue

        full_text = extract_text_from_pdf(pdf_path)

        if full_text:
            batch_updates.append((full_text, paper_id))
            progress["extracted"].append(paper_id)
            extracted += 1
        else:
            progress["failed_extract"].append(paper_id)
            failed_extract += 1

        # Batch update SQLite every 500 papers
        if len(batch_updates) >= 500:
            conn = db.get_connection()
            conn.executemany(
                "UPDATE papers SET full_text = ? WHERE id = ?",
                batch_updates,
            )
            conn.commit()
            conn.close()
            logger.info(
                "Extract progress: %d/%d (extracted=%d, failed=%d) — saved %d to DB",
                i + 1, len(to_extract), extracted, failed_extract, len(batch_updates),
            )
            batch_updates = []
            save_progress(progress)

    # Final batch
    if batch_updates:
        conn = db.get_connection()
        conn.executemany(
            "UPDATE papers SET full_text = ? WHERE id = ?",
            batch_updates,
        )
        conn.commit()
        conn.close()

    save_progress(progress)

    logger.info("=== Extraction complete: %d extracted, %d failed ===",
                 extracted, failed_extract)

    # ── Summary ───────────────────────────────────────────────────────
    conn = db.get_connection()
    total = conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
    with_ft = conn.execute(
        "SELECT COUNT(*) FROM papers WHERE full_text IS NOT NULL AND full_text != ''"
    ).fetchone()[0]
    avg_len = conn.execute(
        "SELECT AVG(LENGTH(full_text)) FROM papers WHERE full_text IS NOT NULL AND full_text != ''"
    ).fetchone()[0]
    conn.close()

    logger.info("=== Summary ===")
    logger.info("Total papers: %d", total)
    logger.info("With full text: %d (%.1f%%)", with_ft, 100 * with_ft / total if total else 0)
    logger.info("Average full text length: %s chars", f"{avg_len:,.0f}" if avg_len else "N/A")


if __name__ == "__main__":
    main()
