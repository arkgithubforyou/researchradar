# ResearchRadar Developer Guide

A production RAG (Retrieval-Augmented Generation) system for NLP/ML research papers. Indexes 26,544 papers from the ACL Anthology with full-text extraction, hybrid retrieval (BM25 + vector + cross-encoder reranking), LLM-powered answers, entity enrichment, and analytics.

**Live demo**: https://thearkforyou-researchradar.hf.space

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| **Backend** | Python 3.12, FastAPI, uvicorn |
| **Frontend** | React 19, TypeScript, Vite, Tailwind CSS, Recharts |
| **Vector DB** | ChromaDB 0.4.24 (cosine distance, 768-dim) |
| **Embeddings** | BAAI/bge-base-en-v1.5 (SentenceTransformers) |
| **Reranker** | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| **Sparse Search** | BM25Okapi (rank-bm25) |
| **LLM** | Groq API (llama-3.3-70b) or local Ollama (qwen2.5:14b) |
| **SQL Store** | SQLite (hand-written SQL, no ORM) |
| **Deployment** | Docker, HuggingFace Spaces, AWS ECS, Oracle Cloud |

---

## Directory Structure

```
researchradar/
├── src/
│   ├── api/                      # FastAPI backend (17 endpoints)
│   │   ├── app.py               # App factory, lifespan events, SPA static serving
│   │   ├── deps.py              # Dependency injection singletons
│   │   ├── models.py            # 18 Pydantic request/response schemas
│   │   ├── routes_search.py     # POST /api/search (RAG pipeline)
│   │   ├── routes_papers.py     # GET /api/papers, /api/papers/{id}
│   │   ├── routes_analytics.py  # Trends, rankings, co-occurrence, growth
│   │   └── rate_limit.py        # Token-bucket rate limiter (30 req/min)
│   ├── ingestion/                # Data loading & chunking
│   │   ├── base_loader.py       # PaperRecord dataclass, DataLoader ABC
│   │   ├── load_acl_anthology.py# ACL Anthology package loader
│   │   ├── load_hf_data.py      # HuggingFace parquet loader
│   │   ├── chunking.py          # 3 strategies: abstract-only, fixed-size, section-aware
│   │   └── embeddings.py        # SentenceTransformer encode + ChromaDB store
│   ├── retrieval/                # Hybrid search pipeline
│   │   ├── bm25_retriever.py    # Sparse retrieval with NLTK tokenization
│   │   ├── vector_retriever.py  # Dense retrieval via ChromaDB
│   │   ├── hybrid_retriever.py  # Reciprocal Rank Fusion (k=60)
│   │   ├── reranker.py          # Cross-encoder reranking
│   │   └── pipeline.py          # Orchestrates: BM25 + vector → RRF → rerank
│   ├── generation/               # LLM backends & RAG engine
│   │   ├── llm_backend_base.py  # LLMBackend ABC, GenerationConfig
│   │   ├── groq_backend.py      # Groq cloud API (fast, needs API key)
│   │   ├── ollama_backend.py    # Local Ollama REST API
│   │   └── rag_engine.py        # Query → retrieve → format context → LLM → answer
│   ├── enrichment/               # Entity extraction
│   │   ├── regex_extractor.py   # Pattern-based: methods, datasets, tasks, topics
│   │   ├── llm_extractor.py     # LLM-powered entity extraction
│   │   └── pipeline.py          # LLM + regex merge, Groq→Ollama auto-failover
│   ├── evaluation/               # Eval framework
│   │   ├── metrics.py           # P@K, R@K, MRR, NDCG, ROUGE-L, answer coverage
│   │   ├── dataset.py           # Eval question loader
│   │   ├── runner.py            # Evaluation orchestrator
│   │   └── ablation.py          # Ablation study framework
│   ├── storage/                  # Data persistence
│   │   ├── sqlite_db.py         # Schema, CRUD, analytics queries
│   │   ├── chroma_store.py      # ChromaDB collection wrapper
│   │   └── s3_store.py          # AWS S3 backup (optional)
│   └── config.py                # Dataclass config from env vars
├── frontend/
│   ├── src/
│   │   ├── pages/               # Route pages
│   │   │   ├── SearchPage.tsx   # RAG search UI with source citations
│   │   │   ├── BrowsePage.tsx   # Paper listing with filters
│   │   │   ├── PaperPage.tsx    # Paper detail (authors, entities)
│   │   │   └── DashboardPage.tsx# Analytics: growth, trends, rankings
│   │   ├── components/          # Shared React components
│   │   │   ├── Layout.tsx       # Nav bar + route outlet
│   │   │   ├── GrowthChart.tsx  # Year-over-year growth chart
│   │   │   ├── VenueChart.tsx   # Papers per venue bar chart
│   │   │   ├── TopEntitiesChart.tsx # Top methods/datasets/tasks
│   │   │   ├── TrendExplorer.tsx# Entity trend visualization
│   │   │   └── CooccurrenceTable.tsx # Method-dataset co-occurrence
│   │   └── lib/
│   │       ├── api.ts           # Fetch wrapper for all API endpoints
│   │       ├── types.ts         # TypeScript types (mirrors Pydantic models)
│   │       └── hooks.ts         # Custom React hooks
│   └── vite.config.ts
├── scripts/                      # CLI tools
│   ├── ingest.py                # Full pipeline: load → chunk → embed
│   ├── enrich.py                # Entity extraction (regex or LLM)
│   ├── setup_db.py              # Create schema + seed data
│   ├── embed_chunks.py          # Memory-safe GPU embedding (mega-batch)
│   ├── rechunk.py               # Re-chunk existing papers
│   ├── extract_fulltext.py      # PyMuPDF PDF → text
│   ├── fulltext_pipeline.py     # GROBID TEI-XML extraction
│   ├── download_pdfs.py         # Bulk PDF downloader
│   ├── grobid_watchdog.py       # GROBID process watchdog with auto-restart
│   ├── write_questions.py       # Interactive eval question authoring
│   ├── annotate.py              # Retrieval relevance annotation
│   ├── annotate_generation.py   # Generation quality scoring
│   ├── retrieval_eval.py        # Per-method retrieval metrics
│   └── run_ablation.py          # Full ablation study
├── Dockerfile                    # Local/AWS: 3-stage (frontend, deps, runtime)
├── Dockerfile.hf                 # HF Spaces: 4-stage (adds data download)
├── docker-compose.yml            # API + Ollama + workers
├── requirements.txt              # Python deps (chromadb==0.4.24, numpy<2.0)
└── data/                         # Runtime data (not in git)
    ├── researchradar.db          # SQLite (26K papers, 394K chunks, enrichment)
    ├── chroma_db/                # ChromaDB (394K embeddings)
    └── raw/                      # PDFs, GROBID output (local pipeline only)
```

---

## Quick Start (Local Development)

### Prerequisites
- Python 3.12+
- Node.js 22+
- (Optional) Ollama for local LLM
- (Optional) NVIDIA GPU for fast embedding

### Setup

```bash
# Clone
git clone https://github.com/arkgithubforyou/researchradar.git
cd researchradar

# Python environment
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Environment
cp .env.example .env
# Edit .env: set GROQ_API_KEY if using Groq, or start Ollama

# Frontend deps
cd frontend && npm install && cd ..
```

### Ingest Data

```bash
# Small test set (100 papers, abstract-only chunks)
python scripts/ingest.py --source acl --year-from 2024 --year-to 2025 \
  --venues acl emnlp --max-papers 100

# Full ingest (all major venues, 2020-2025)
python scripts/ingest.py --source acl --year-from 2020 --year-to 2025 \
  --venues acl emnlp naacl eacl findings --chunk-strategy abstract

# Enrich with entity extraction
python scripts/enrich.py --mode regex --show-stats
```

### Run

```bash
# Backend (port 8000)
uvicorn src.api.app:app --reload

# Frontend dev server (port 5173, proxies to backend)
cd frontend && npm run dev
```

Open http://localhost:5173

---

## Configuration

All config is via environment variables, loaded through `src/config.py`:

| Variable | Default | Description |
|----------|---------|-------------|
| `GROQ_API_KEY` | (empty) | Groq API key for cloud LLM |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama server URL |
| `LLM_BACKEND` | `ollama` | `ollama` or `groq` |
| `EMBEDDING_MODEL` | `BAAI/bge-base-en-v1.5` | SentenceTransformer model |
| `RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder model |
| `SQLITE_DB_PATH` | `./data/researchradar.db` | SQLite database path |
| `CHROMA_DB_PATH` | `./data/chroma_db` | ChromaDB directory path |
| `AWS_S3_BUCKET` | `researchradar-data` | S3 bucket for snapshots |

---

## API Reference

### Health
- `GET /api/health` → `{status, paper_count, chunk_count}`

### Search (RAG)
- `POST /api/search` → RAG pipeline: retrieve + generate
  - Body: `{query, top_k?, year_min?, year_max?, venue?}`
  - Response: `{answer, sources[], model, usage}`
  - Rate-limited: 30 req/min per IP

### Papers
- `GET /api/papers` → Browse with filters
  - Params: `venue, year, method, dataset, author, limit, offset`
  - Response: `{papers[], count, limit, offset}`
- `GET /api/papers/{paper_id}` → Paper detail
  - Note: `paper_id` contains `::` (e.g., `acl_anthology::2024.acl-long.1`)

### Analytics

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/analytics/stats` | GET | Enrichment stats (unique counts per entity type) |
| `/api/analytics/growth` | GET | Year-over-year paper growth |
| `/api/analytics/venues` | GET | Papers per venue per year |
| `/api/analytics/venues-total` | GET | Total papers per venue |
| `/api/analytics/{type}/top` | GET | Top-N entities (type: methods/datasets/tasks/topics) |
| `/api/analytics/{type}/list` | GET | All unique entities with paper counts (limit param) |
| `/api/analytics/{type}/trend` | POST | Entity adoption trend over time |
| `/api/analytics/cooccurrence/method-dataset` | GET | Method-dataset co-occurrence |
| `/api/analytics/cooccurrence/method-task` | GET | Method-task co-occurrence |

---

## Architecture Deep Dive

### RAG Pipeline Flow

```
User Query
    │
    ├─── BM25 Retriever ──────────┐
    │    (sparse, keyword match)   │
    │                              ├── Reciprocal Rank Fusion ── Cross-Encoder ── Top-K
    └─── Vector Retriever ────────┘        (k=60)              Reranking
         (dense, semantic match)
                                                                    │
                                                                    ▼
                                                            Format Context
                                                            (markdown + citations)
                                                                    │
                                                                    ▼
                                                            LLM Generation
                                                            (Groq or Ollama)
                                                                    │
                                                                    ▼
                                                            Answer + Sources
```

### Retrieval Config (tunable)

```python
bm25_top_k = 100       # BM25 candidate pool
vector_top_k = 100     # Vector candidate pool
rrf_k = 60             # RRF smoothing constant
bm25_weight = 1        # RRF weight for BM25
vector_weight = 1      # RRF weight for vector
hybrid_top_k = 50      # Post-fusion pool size
rerank_top_k = 20      # Final results after cross-encoder
```

### Chunking Strategies

| Strategy | Tokens/Chunk | Overlap | Use Case |
|----------|-------------|---------|----------|
| `abstract` | Variable | 0 | Default: 1 chunk per paper (abstract only) |
| `fixed` | 512 | 50 | Full-text papers, maximum retrieval depth |
| `section` | Variable | 0 | GROBID-extracted papers, section-aware |

Current production data uses `fixed` strategy: 26,544 papers → 394,777 chunks.

### Paper ID Format

All paper IDs follow: `{source}::{source_id}`
- Example: `acl_anthology::2024.acl-long.1`
- The `::` separator is important for URL routing (use `:path` parameter)

---

## SQLite Schema

```sql
-- Papers table
CREATE TABLE papers (
    id TEXT PRIMARY KEY,           -- "source::source_id"
    source_id TEXT, source TEXT,
    title TEXT, abstract TEXT, full_text TEXT,
    year INTEGER, venue TEXT, volume TEXT, url TEXT,
    created_at TIMESTAMP
);

-- Authors (ordered)
CREATE TABLE authors (
    id INTEGER PRIMARY KEY, paper_id TEXT, name TEXT, position INTEGER
);

-- Text chunks for retrieval
CREATE TABLE chunks (
    id INTEGER PRIMARY KEY, paper_id TEXT,
    chunk_text TEXT, chunk_type TEXT, chunk_index INTEGER, token_count INTEGER
);

-- Enrichment entities (one table per type)
CREATE TABLE methods (id, paper_id, method_name);
CREATE TABLE datasets (id, paper_id, dataset_name, type);
CREATE TABLE tasks (id, paper_id, task_name);
CREATE TABLE topics (id, paper_id, topic_name);
```

All queries use parameterized SQL (no ORM, no f-strings). Analytics queries are in `sqlite_db.py`.

---

## Full-Text Pipeline (Offline)

The full-text extraction pipeline runs locally (not in Docker/HF):

```
1. Download PDFs          → data/raw/pdfs/           (scripts/download_pdfs.py)
2. PyMuPDF extraction     → data/raw/pymupdf/        (scripts/extract_fulltext.py)
3. GROBID TEI-XML         → data/raw/grobid/         (scripts/fulltext_pipeline.py)
4. Re-chunk               → SQLite chunks table       (scripts/rechunk.py)
5. Embed                  → ChromaDB                  (scripts/embed_chunks.py)
6. Enrich                 → SQLite entity tables      (scripts/enrich.py)
7. Upload to HF dataset   → huggingface_hub API
8. Push code to HF Space  → git push / HF API
```

### GPU Embedding (embed_chunks.py)

For large-scale embedding on a GPU machine:

```bash
python scripts/embed_chunks.py \
  --mega-batch 10000 \     # Process 10K chunks at a time
  --encode-batch 128 \     # GPU encoding batch size
  --chroma-batch 500       # ChromaDB insertion batch size
```

The mega-batch approach avoids OOM: encode 10K → store → gc.collect() → next 10K.

**Performance** (RTX 4090): ~300 chunks/sec, 394K chunks in ~25 minutes.

---

## Deployment

### HuggingFace Spaces (Current Production)

**Architecture**: Pre-built data pattern
1. Data is ingested locally (GPU machine for embeddings)
2. SQLite + ChromaDB uploaded to HF dataset repo (`thearkforyou/researchradar-data`)
3. `Dockerfile.hf` downloads data at Docker build time via `huggingface_hub.snapshot_download()`
4. Models (embedding + cross-encoder) are cached at build time

**To redeploy after data changes:**

```bash
# 1. Upload new data to dataset repo
python -c "
from huggingface_hub import HfApi
api = HfApi(token='YOUR_HF_TOKEN')

# Upload SQLite
api.upload_file(
    path_or_fileobj='data/researchradar.db',
    path_in_repo='researchradar.db',
    repo_id='thearkforyou/researchradar-data',
    repo_type='dataset',
)

# Upload ChromaDB
api.upload_folder(
    folder_path='data/chroma_db',
    path_in_repo='chroma_db',
    repo_id='thearkforyou/researchradar-data',
    repo_type='dataset',
)
"

# 2. Push code changes to trigger rebuild
# Either via git push to HF remote, or via API:
python -c "
from huggingface_hub import HfApi, CommitOperationAdd
api = HfApi(token='YOUR_HF_TOKEN')
api.create_commit(
    repo_id='thearkforyou/researchradar',
    repo_type='space',
    operations=[CommitOperationAdd(path_in_repo='requirements.txt', path_or_fileobj='requirements.txt')],
    commit_message='Trigger rebuild',
)
"
```

**HF Space specs** (free tier): 2 vCPU, 16 GB RAM, 50 GB ephemeral disk.

**Critical version pins**: `chromadb==0.4.24` and `numpy<2.0` must match the data format.

### Docker Compose (Self-Hosted)

```bash
# Start API + Ollama
docker compose up -d

# Initial data setup
docker compose --profile setup run ingest

# Enrichment
docker compose --profile tools run enrich

# Pull LLM model
docker compose exec ollama ollama pull qwen2.5:14b
```

### Environment Setup for HF Spaces

Set these as Space secrets:
- `GROQ_API_KEY`: Your Groq API key (required for RAG search)
- `LLM_BACKEND`: `groq` (Ollama not available on HF free tier)

---

## CI/CD

`.github/workflows/ci.yml` — On push/PR to `main`: lint (ruff + black) → test (pytest 316 tests).

`.github/workflows/deploy.yml` — On merge to `main`: build image → push to ECR (tagged SHA + latest) → update ECS task def → deploy to Fargate. Requires `AWS_DEPLOY_ROLE_ARN` secret (OIDC, no long-lived keys).

`.github/workflows/keep-alive.yml` — Cron every 12h, pings HF Space `/api/health` to prevent 48h sleep. Needs `HF_SPACE_URL` repo variable.

---

## Resource Usage & Scaling

### Current Data Sizes (measured Feb 2026)

| Component | Size | Notes |
|-----------|------|-------|
| **SQLite DB** | 2.39 GB | 26,544 papers, 394,777 chunks, enrichment tables |
| **ChromaDB total** | 8.56 GB | Embeddings only (no document text stored) |
| — chroma.sqlite3 | 7.54 GB | Redundant: stores embeddings in SQLite (backup) |
| — data_level0.bin (HNSW) | 1.21 GB | Actual vector index used for queries |
| — other (metadata, links) | 17 MB | Pickle, link lists, header |
| **Total deployed** | ~10.95 GB | Downloaded at Docker build time |

**Per-chunk storage**: ~29 KB (22.7 KB ChromaDB + 6.3 KB SQLite)

**Dataset counts**: 26,544 papers | 394,777 chunks | 25,122 methods | 3,402 datasets | 16,312 tasks | 38,709 topics | avg 494.6 tokens/chunk | 923 MB total chunk text

### ChromaDB Redundancy Problem

chromadb 0.4.24 stores every 768-dim embedding vector twice:
1. **chroma.sqlite3** (7.54 GB) — SQLite persistence/recovery copy
2. **data_level0.bin** (1.21 GB) — HNSW binary index for actual queries

Only the HNSW index is needed at query time. The SQLite copy is an architectural decision in chromadb 0.4.x that cannot be disabled. This is why removing document text (done previously) only saved 0.32 GB — the bulk is embedding data.

**Fix**: Migrate to FAISS, which stores only the index file (~1.2 GB). See Future TODOs.

### HF Free Tier Budget

| Resource | Limit | Current Usage | Used % | Headroom |
|----------|-------|---------------|--------|----------|
| **Disk** | 50 GB | ~28 GB (data + Docker layers + models) | 56% | ~22 GB |
| **RAM** | 16 GB | ~5.5-6 GB peak | 37% | ~10 GB |
| **Build time** | ~30 min | ~30-35 min | ~100% | Almost none |

### RAM Breakdown (at startup, after first query)

| Component | RAM | Scales with data? |
|-----------|-----|-------------------|
| Embedding model (bge-base-en-v1.5) | ~600 MB | No |
| Cross-encoder reranker | ~200 MB | No |
| BM25 index (tokenized chunks in memory) | ~800 MB-1 GB | **Yes, linear** |
| Chunk lookup cache (`_chunk_map`, lazy) | ~500 MB-1 GB | **Yes, linear** |
| ChromaDB client + HNSW mmap | ~200 MB | Partly |
| Python + FastAPI + misc | ~500 MB | No |

### Docker Build Time Breakdown

| Stage | Time | Scales with data? |
|-------|------|-------------------|
| Stage 1: npm ci + frontend build | ~3 min | No |
| Stage 2: pip install deps | ~5 min | No |
| **Stage 3: `snapshot_download()` ~10.9 GB** | **~10-15 min** | **Yes** |
| Stage 4: Model download + NLTK | ~10 min | No |
| **Total** | **~30-35 min** | |

**Important**: HF Spaces has two separate timeouts:
- `startup_duration_timeout` (configurable in README.md YAML, ours = 30m) — runtime startup
- Docker build timeout (~30 min, NOT configurable) — how long HF allows the image to build

### Scaling Projections

| Scale | Papers | Chunks | Disk | RAM | Build Time | Fits free tier? |
|-------|--------|--------|------|-----|------------|-----------------|
| **1× (current)** | 26.5K | 394K | ~28 GB | ~5.5 GB | ~30-35 min | Barely |
| **2×** | ~53K | ~790K | ~39 GB | ~8-9 GB | ~45 min | Build timeout ❌ |
| **3×** | ~80K | ~1.18M | ~50 GB | ~11-13 GB | ~55 min | Disk + build ❌ |

**Bottleneck order**: Build timeout → Disk → RAM

### With FAISS Migration (projected)

| Scale | Data Download | Build Time | Disk | Fits? |
|-------|-------------|------------|------|-------|
| **1×** | ~3.6 GB | ~22 min | ~21 GB | Yes ✅ |
| **2×** | ~7.2 GB | ~28 min | ~32 GB | Yes ✅ |
| **3×** | ~10.8 GB | ~33 min | ~43 GB | Tight ⚠️ |

---

## Enrichment

Entity extraction populates methods/datasets/tasks/topics tables:

```bash
# Fast regex-only extraction
python scripts/enrich.py --mode regex --show-stats

# LLM extraction with auto-failover (Groq → Ollama)
python scripts/enrich.py --mode llm --backend groq --show-stats

# Limit for testing
python scripts/enrich.py --mode regex --max-papers 100
```

**Current production stats** (26,544 papers):
- 25,122 methods | 3,402 datasets | 16,312 tasks | 38,709 topics
- 12,778 papers with at least one method extracted
- Top methods (2025): fine-tuning (1127), GPT (662), zero-shot (443), RAG (439)

---

## Evaluation

### Create Ground Truth

```bash
# 1. Author eval questions
python scripts/write_questions.py

# 2. Annotate retrieval relevance
python scripts/annotate.py

# 3. Score generation quality
python scripts/annotate_generation.py
```

### Run Evaluations

```bash
# Per-method retrieval metrics (P@K, R@K, MRR, NDCG)
python scripts/retrieval_eval.py

# Full ablation study
python scripts/run_ablation.py
```

---

## Code Conventions

- **Type hints everywhere** — use `X | None` (Python 3.10+ style)
- **No ORM** — hand-written parameterized SQL queries
- **DI pattern** — singletons created at FastAPI lifespan, injected via `Depends()`
- **Logging** — `logging.getLogger(__name__)`, never `print()`
- **Config** — all via `.env` → dataclass, no hardcoded values
- **Paper IDs** — always `{source}::{source_id}` format

---

## Troubleshooting

### ChromaDB version mismatch
The ChromaDB data format is version-specific. Local data was created with `chromadb==0.4.24`. If you see `KeyError: '_type'` or similar, ensure the exact version is installed.

### numpy compatibility
`chromadb==0.4.24` uses `np.float_` which was removed in NumPy 2.0. Pin `numpy<2.0` in requirements.

### ONNX Runtime DLL errors (Windows)
If you see `DLL initialization routine failed` for onnxruntime, downgrade to `onnxruntime==1.20.1`. PyTorch CUDA installs can break the ONNX DLL.

### chroma-hnswlib segfaults (Windows)
`chroma-hnswlib==0.7.6` (paired with chromadb 0.5.x) causes segfaults on Windows. Use `chromadb==0.4.24` with `chroma-hnswlib==0.7.3`.

### BM25 index build slow
The BM25 index is built from all chunks at startup. With 394K chunks, this takes ~30-60s on 2 vCPU. The health check has a 120s start-period to accommodate this.

### HF Space Docker build timeout
The Docker build timeout on HF Spaces is ~30 min (NOT configurable — separate from `startup_duration_timeout` which only controls runtime startup). Our build is currently at ~30-35 min, with data download (`snapshot_download` in Stage 3) as the bottleneck. Scaling data size will push past this limit. See "Resource Usage & Scaling" section for projections.

### HF Space build fails with disk space
The data download (~11 GB) + Docker layers need ~25-30 GB during build. The free tier has 50 GB, which is sufficient but tight.

---

## Package Version Notes (Critical)

These versions are tested and compatible with each other:

```
chromadb==0.4.24          # MUST match data format
chroma-hnswlib==0.7.3     # Stable on Windows + Linux
numpy>=1.24.0,<2.0        # chromadb 0.4.x uses deprecated np.float_
sentence-transformers>=2.2.0
onnxruntime==1.20.1       # For Windows CUDA environments
```

---

## Repository Links

- **Source code**: https://github.com/arkgithubforyou/researchradar
- **HF Space**: https://huggingface.co/spaces/thearkforyou/researchradar
- **HF Dataset**: https://huggingface.co/datasets/thearkforyou/researchradar-data
- **Live demo**: https://thearkforyou-researchradar.hf.space
