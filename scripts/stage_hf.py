"""Stage files for Hugging Face Spaces upload."""

import shutil
from pathlib import Path

SRC = Path("E:/RAG/researchradar")
DST = Path("/tmp/hf-deploy")

# Clean staging
if DST.exists():
    shutil.rmtree(DST)
DST.mkdir(parents=True)

EXCLUDE_DIRS = {
    ".git", ".github", "data", ".venv", "venv", "__pycache__",
    "terraform", "terraform-oci", ".pytest_cache", ".ruff_cache",
    "htmlcov", "node_modules", "dist", ".idea", "tests",
    ".vite", ".cache",
}

EXCLUDE_FILES = {
    ".env", "Thumbs.db", ".DS_Store", "CLAUDE.md",
    "Dockerfile", "Dockerfile.hf", ".dockerignore",
    "tsconfig.tsbuildinfo",
}

EXCLUDE_SUFFIXES = {".pyc", ".pyo", ".log", ".pem"}

count = 0
for f in SRC.rglob("*"):
    if not f.is_file():
        continue

    # Skip excluded directories
    if any(part in EXCLUDE_DIRS for part in f.relative_to(SRC).parts):
        continue

    # Skip excluded files
    if f.name in EXCLUDE_FILES:
        continue

    # Skip excluded suffixes
    if f.suffix in EXCLUDE_SUFFIXES:
        continue

    rel = f.relative_to(SRC)
    dst_path = DST / rel
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(f, dst_path)
    count += 1

# Copy Dockerfile.hf as Dockerfile
shutil.copy2(SRC / "Dockerfile.hf", DST / "Dockerfile")
count += 1

# Copy .dockerignore
shutil.copy2(SRC / ".dockerignore", DST / ".dockerignore")
count += 1

# Create README.md with HF frontmatter
readme = DST / "README.md"
readme.write_text("""\
---
title: ResearchRadar
emoji: "\U0001F52C"
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
short_description: RAG-powered NLP research paper explorer
startup_duration_timeout: 30m
pinned: false
---

# ResearchRadar

RAG-powered NLP/ML research paper explorer with hybrid retrieval and LLM generation.
""", encoding="utf-8")
count += 1

print(f"Staged {count} files")

for p in sorted(DST.rglob("*")):
    if p.is_file():
        rel = p.relative_to(DST)
        print(f"  {rel} ({p.stat().st_size:,} bytes)")
