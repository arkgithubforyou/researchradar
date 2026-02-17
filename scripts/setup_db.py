"""CLI: Initialize the SQLite database schema.

Usage:
    python scripts/setup_db.py
    python scripts/setup_db.py --db-path ./data/researchradar.db
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import get_config
from src.storage.sqlite_db import SQLiteDB


def main():
    parser = argparse.ArgumentParser(description="Initialize ResearchRadar SQLite database")
    parser.add_argument("--db-path", type=str, default=None, help="Override database path")
    args = parser.parse_args()

    config = get_config()
    db_path = args.db_path or config.sqlite_db_path

    db = SQLiteDB(db_path)
    db.create_schema()
    print(f"Database initialized at: {db_path}")


if __name__ == "__main__":
    main()
