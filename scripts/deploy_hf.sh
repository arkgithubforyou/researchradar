#!/bin/bash
# Deploy ResearchRadar to Hugging Face Spaces.
#
# Prerequisites:
#   1. Create a Space at huggingface.co/new-space (Docker SDK)
#   2. Set GROQ_API_KEY as a Secret in Space Settings
#   3. Install: pip install huggingface_hub
#   4. Login: huggingface-cli login
#
# Usage:
#   ./scripts/deploy_hf.sh <hf_username>/<space_name>
#
# Example:
#   ./scripts/deploy_hf.sh arkgithubforyou/researchradar

set -e

SPACE_ID="${1:?Usage: deploy_hf.sh <hf_username>/<space_name>}"
SPACE_REPO="https://huggingface.co/spaces/${SPACE_ID}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "=== Deploying ResearchRadar to HF Space: ${SPACE_ID} ==="

# Create a temp directory for the HF Space repo
TMPDIR=$(mktemp -d)
trap "rm -rf $TMPDIR" EXIT

echo "1. Cloning HF Space repo..."
git clone "${SPACE_REPO}" "$TMPDIR/space" 2>/dev/null || {
    echo "   Space doesn't exist yet or clone failed."
    echo "   Create it first at: https://huggingface.co/new-space"
    echo "   - Select 'Docker' as the SDK"
    echo "   - Set hardware to 'CPU basic (Free)'"
    exit 1
}

echo "2. Copying project files..."
# Copy everything needed for the Docker build
rsync -a --delete \
    --exclude='.git/' \
    --exclude='data/' \
    --exclude='.venv/' \
    --exclude='venv/' \
    --exclude='__pycache__/' \
    --exclude='*.pyc' \
    --exclude='.env' \
    --exclude='terraform/' \
    --exclude='terraform-oci/' \
    --exclude='frontend/node_modules/' \
    --exclude='frontend/dist/' \
    --exclude='.pytest_cache/' \
    --exclude='.ruff_cache/' \
    --exclude='htmlcov/' \
    "$PROJECT_DIR/" "$TMPDIR/space/"

echo "3. Setting up HF Space config..."
# Use the HF-specific Dockerfile
cp "$TMPDIR/space/Dockerfile.hf" "$TMPDIR/space/Dockerfile"

# Create the required README.md with YAML frontmatter
cat > "$TMPDIR/space/README.md" << 'EOF'
---
title: ResearchRadar
emoji: 🔬
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
short_description: RAG-powered NLP research paper explorer
startup_duration_timeout: 30m
pinned: false
---

# ResearchRadar

RAG-powered NLP/ML research paper explorer. Hybrid retrieval (BM25 + vector + cross-encoder reranking) + LLM generation over ACL Anthology papers.

**Features:**
- 🔍 Ask natural language questions about NLP research
- 📄 Browse and filter papers by venue, year, method, dataset
- 📊 Interactive analytics dashboard with trends, top entities, and co-occurrence
- ⚡ Powered by Groq API for fast LLM inference

Built with FastAPI, React, ChromaDB, and sentence-transformers.
EOF

echo "4. Pushing to HF Space..."
cd "$TMPDIR/space"
git add -A
git commit -m "Deploy ResearchRadar $(date +%Y-%m-%d)" 2>/dev/null || {
    echo "   No changes to deploy."
    exit 0
}
git push

echo ""
echo "=== Deployed! ==="
echo "Space URL: ${SPACE_REPO}"
echo ""
echo "NOTE: First build takes ~5-10 minutes."
echo "Set GROQ_API_KEY as a Secret in Space Settings:"
echo "  ${SPACE_REPO}/settings"
