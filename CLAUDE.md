# ResearchRadar

RAG system for NLP/ML research papers. Hybrid retrieval (vector + BM25 + cross-encoder reranking), LLM generation, SQL trend analytics, FastAPI backend.

## Stack

Python 3.12 | SQLite (hand-written SQL, no ORM) | ChromaDB | FastAPI
Frontend: React 19 + TypeScript + Vite + Tailwind CSS + Recharts
Embeddings: BAAI/bge-base-en-v1.5 | Reranker: ms-marco-MiniLM-L-6-v2 | BM25: rank-bm25
LLM: Ollama/Qwen2.5-14B local, Groq API cloud | AWS S3 (DB snapshots)

## Conventions

- Type hints everywhere. Ruff + Black.
- Hand-written parameterized SQL (no ORM). No f-strings for SQL.
- ABCs for extensibility (`DataLoader` in `base_loader.py`, `LLMBackend` in `llm_backend_base.py`).
- Validate at system boundaries only. Log errors with context.
- pytest with small fixtures + mocks for expensive components. `pytest tests/ -v`
- Config via `python-dotenv` → `src/config.py` dataclass. Env vars in `.env.example`.
- Paper IDs: `{source}::{source_id}` (e.g., `hf_acl_ocl::P18-1001`)

## Modules

- `src/ingestion/` — `base_loader.py` (PaperRecord, DataLoader ABC), `chunking.py`, `embeddings.py`, loaders
- `src/storage/` — `sqlite_db.py`, `chroma_store.py`, `s3_store.py`
- `src/retrieval/` — `bm25_retriever.py`, `vector_retriever.py`, `hybrid_retriever.py`, `reranker.py`, `pipeline.py`
- `src/generation/` — `llm_backend_base.py` (LLMBackend ABC), `ollama_backend.py`, `groq_backend.py`, `rag_engine.py`
- `src/enrichment/` — `regex_extractor.py`, `llm_extractor.py`, `pipeline.py` (LLM+regex entity extraction)
- `src/api/` — `app.py` (FastAPI factory + lifespan + SPA serving), `deps.py` (DI singletons), `models.py` (18 Pydantic schemas), `routes_search.py` (rate-limited), `routes_papers.py`, `routes_analytics.py`, `rate_limit.py`
- `frontend/` — React SPA: `src/pages/` (Search, Browse, Paper, Dashboard), `src/components/` (Layout, charts, tables), `src/lib/` (api.ts, types.ts, hooks.ts)
- `src/evaluation/` — `metrics.py` (P@K, R@K, MRR, NDCG, ROUGE-L), `dataset.py` (eval dataset loader), `runner.py` (eval runner), `ablation.py` (ablation study framework)
- `scripts/` — `ingest.py`, `enrich.py`, `smoke_test_llm.py`, `write_questions.py` (question authoring), `annotate.py` (TREC-style retrieval annotation), `annotate_generation.py` (generation scoring), `retrieval_eval.py` (per-method metrics), `run_ablation.py` (full ablation runner)
- `data/` — `questions.json` (authored questions), `eval_set.json` (annotated ground truth), `researchradar.db`, `chroma_db/`, `raw/`

## Progress

- [x] Phase 1: Loaders, SQLite, chunking, ChromaDB, embeddings, S3, ingest CLI
- [x] Phase 2: BM25, vector, hybrid RRF, cross-encoder reranker, retrieval pipeline
- [x] Phase 3: LLM backend ABC, Ollama + Groq backends, RAG engine, smoke test
- [x] Phase 4: LLM entity extraction, regex fallback, SQL analytics
- [x] Phase 5: FastAPI endpoints (17 routes), eval framework, annotation pipeline, ablation study
- [x] Phase 6: Docker Compose, deploy, CI/CD, Terraform
- [x] Phase 7: React frontend (Search, Browse, Paper Detail, Dashboard), static serving, rate limiting, HTTPS Terraform
- [ ] Phase 8: Full-text extraction (IN PROGRESS — see below)

## API

`uvicorn src.api.app:app` — 17 endpoints under `/api`:

| Endpoint | Purpose |
|---|---|
| `GET /api/health` | Health + paper/chunk counts |
| `POST /api/search` | RAG search with year/venue filters |
| `GET /api/papers` | Browse with filters |
| `GET /api/papers/{id}` | Paper detail (`:path` for `::` IDs) |
| `POST /api/analytics/{type}/trend` | Trend by entity (methods/datasets/tasks/topics) |
| `GET /api/analytics/{type}/top` | Top-N ranking (methods/datasets/tasks/topics) |
| `GET /api/analytics/cooccurrence/{type}` | Co-occurrence (method-dataset, method-task) |
| `GET /api/analytics/stats` | Enrichment stats |
| `GET /api/analytics/growth` | Papers per year |
| `GET /api/analytics/venues` | Papers per venue |

DI via module singletons in `deps.py`. Lifespan loads models once at startup; `is_initialized()` guard skips loading during tests.

## Data Plan

**Initial corpus**: ACL Anthology (via `acl-anthology` package), scoped to major venues 2020-2025.
- Source: `--source acl --year-from 2020 --year-to 2025 --venues acl emnlp naacl findings`
- Expected: ~5-10K papers (abstracts only, no full text)
- Chunking: `abstract` strategy (1 chunk per paper)
- Embeddings: BAAI/bge-base-en-v1.5, ~10-20 min on CPU
- Enrichment: `--mode llm --backend groq` with auto-fallback to Ollama if Groq quota exhausted
- Groq free tier handles ~14K requests/day — should cover the full initial corpus in one run

**Commands**:
```bash
python scripts/ingest.py --source acl --year-from 2020 --year-to 2025 --venues acl emnlp naacl findings
python scripts/enrich.py --mode llm --backend groq --show-stats
```

**Expansion later**: Add older years, more venues (COLING, EACL, TACL, workshops), or the HF parquet (74K papers with full text through 2022).

## Annotation Workflow

Three-step pipeline for building ground truth and running evaluation:

```bash
# Step 1: Author questions
python scripts/write_questions.py            # interactive loop → data/questions.json
python scripts/write_questions.py --list     # review existing questions

# Step 2: Annotate retrieval relevance (TREC-style pooled judging)
python scripts/annotate.py                   # pools BM25+vector+hybrid, judge y/n/s per chunk → data/eval_set.json
python scripts/annotate.py --question "..."  # ad-hoc single question

# Step 3: Score generation quality
python scripts/annotate_generation.py        # runs RAG, score faithfulness/relevance/citations → appended to eval_set.json

# Evaluate
python scripts/retrieval_eval.py             # per-method Hit Rate, MRR, P@K, R@K, NDCG from annotations
python scripts/run_ablation.py               # full ablation: retrieval comparison + generation quality summary
```

## Docker

Three-stage `Dockerfile`: (1) Node.js builds frontend, (2) Python builder installs deps, (3) slim Python 3.12 runtime. Non-root user, NLTK data baked in, frontend dist copied in, 120s health check start period for model loading.

```bash
docker compose up              # API + Ollama
docker compose --profile setup run ingest   # one-shot DB init
docker compose --profile tools run enrich   # one-shot enrichment
```

Services: `api` (FastAPI :8000), `ollama` (LLM :11434), `ingest`/`enrich` (one-shot workers behind profiles). Volumes: `app-data`, `model-cache`, `ollama-models`.

## CI/CD

`.github/workflows/ci.yml` — On push/PR to `main`: lint (ruff + black) → test (pytest 316 tests).

`.github/workflows/deploy.yml` — On merge to `main`: build image → push to ECR (tagged SHA + latest) → update ECS task def → deploy to Fargate. Requires `AWS_DEPLOY_ROLE_ARN` secret (OIDC, no long-lived keys).

`.github/workflows/keep-alive.yml` — Cron every 12h, pings HF Space `/api/health` to prevent 48h sleep. Needs `HF_SPACE_URL` repo variable.

## Infrastructure (Terraform)

Three deployment options:

### Hugging Face Spaces — Free ($0/month, easiest) ← CURRENTLY DEPLOYED

`Dockerfile.hf` + `scripts/stage_hf.py` — Docker Space on HF free tier (2 vCPU, 16 GB RAM).

**Live at**: https://thearkforyou-researchradar.hf.space

**Deployment pattern**: Pre-built data approach (ingest locally → upload to HF dataset → Dockerfile downloads at build time).
- `Dockerfile.hf`: 4-stage build — (1) Node frontend, (2) Python deps, (3) Download pre-built data from HF dataset `thearkforyou/researchradar-data`, (4) Runtime with model pre-download
- `scripts/stage_hf.py`: Stages files for HF upload, creates README.md with HF YAML frontmatter (emoji: 🔬, sdk: docker, port: 7860)
- Pre-downloads `BAAI/bge-base-en-v1.5` and `cross-encoder/ms-marco-MiniLM-L-6-v2` at build time

**Currently deployed**: 26,544 papers (2020-2025), abstract-only (~461 MB). Full-text extraction in progress (see below).

```bash
python scripts/stage_hf.py          # stage files to hf_staging/
cd hf_staging && git push           # push to HF Space
```

### Oracle Cloud — Always Free ($0/month, recommended for demos)

`terraform-oci/` — Oracle Cloud A1 ARM VM. `terraform apply` creates:

- **Compute**: VM.Standard.A1.Flex (2 OCPU, 12 GB RAM — within Always Free limits)
- **Networking**: VCN, public subnet, internet gateway, security list (80/443/22)
- **Auto-setup**: cloud-init installs Docker, Nginx, Let's Encrypt, clones repo, builds and runs container
- **Redeploy**: SSH in, run `/opt/researchradar/redeploy.sh` to pull latest and rebuild

```bash
cd terraform-oci
cp terraform.tfvars.example terraform.tfvars  # fill in your OCI credentials
terraform init && terraform apply
```

### AWS — ECS Fargate (~$100/month, enterprise-grade)

`terraform/` — AWS ECS Fargate deployment. `terraform apply` creates:

- **Networking**: VPC, 2 AZ public/private subnets, NAT gateway, internet gateway
- **Traffic**: ALB (health checks `/api/health`), optional HTTPS via `certificate_arn` var, security groups (ALB→ECS only)
- **Compute**: ECS cluster, Fargate task (1 vCPU, 4 GB), auto-restart on crash
- **Storage**: ECR (image registry, keeps last 10), CloudWatch logs (30-day retention)
- **IAM**: execution role (pull images + logs), task role (S3 `researchradar-data` only)

Config in `variables.tf`, outputs (ALB DNS, ECR URL) in `outputs.tf`. State backend commented out — uncomment for team use.

## HF Credentials

- **Token**: (stored in HF Spaces secrets and local env — do NOT commit)
- **Username**: `thearkforyou`
- **Space**: `thearkforyou/researchradar`
- **Dataset repo**: `thearkforyou/researchradar-data` (holds pre-built SQLite + ChromaDB)
- **GitHub**: `arkgithubforyou/researchradar`

## Full-Text Extraction Pipeline (ONGOING)

### Status
Extracting full text from all 26,544 ACL Anthology papers (currently abstract-only).

### Architecture
Two-phase pipeline running unattended:

1. **Phase A — PyMuPDF extraction** (running NOW)
   - Script: `scripts/extract_fulltext.py` — downloads PDFs + extracts text with PyMuPDF (fitz)
   - Watchdog: `scripts/watchdog.sh` — bash wrapper with auto-restart (MAX_RETRIES=20), launched via `nohup`
   - Progress: tracked in `data/raw/fulltext_progress.json` (downloaded/extracted/failed lists)
   - PDFs stored in `data/raw/pdfs/`, ~1.5 MB each
   - After completion: **auto-reboots** to enable WSL2 (line 97 of watchdog.sh: `shutdown.exe /r /t 60`)

2. **Phase B — GROBID extraction** (starts automatically after reboot)
   - Script: `scripts/fulltext_pipeline.py` — sends PDFs to GROBID Docker container, parses TEI-XML output
   - Watchdog: `scripts/watchdog.ps1` — PowerShell, runs via Windows Scheduled Task `ResearchRadar-Pipeline` (trigger: AtLogon)
   - Flow: checks PyMuPDF done → waits for Docker → pulls `lfoppiano/grobid:0.8.1` → starts container on port 8070 → runs `fulltext_pipeline.py --skip-download`
   - GROBID produces higher quality text (~90-95% vs PyMuPDF's ~85-90%): proper 2-column handling, section detection, header/footer removal
   - Progress: tracked in `data/raw/pipeline_progress.json` (downloaded/grobid_done/grobid_failed/db_updated sets)
   - Extracted text cached in `data/raw/tei/{safe_name}.txt`
   - GROBID URL: `http://localhost:8070`, Docker image: `lfoppiano/grobid:0.8.1`, container name: `grobid`
   - Estimated time: ~22 hours for 26K papers

### Why the reboot?
WSL2 is required for Docker Desktop (Linux engine). Virtual Machine Platform was enabled via `Enable-WindowsOptionalFeature` but requires a reboot to activate. Without it, Docker returns 500 errors.

### Windows Scheduled Tasks
- `ResearchRadar-Pipeline` — runs `watchdog.ps1` on login (trigger: AtLogon, runs as Admin). Handles both PyMuPDF completion check and GROBID pipeline.

### Key files
| File | Purpose |
|------|---------|
| `scripts/extract_fulltext.py` | PyMuPDF download + extraction (Phase A) |
| `scripts/fulltext_pipeline.py` | GROBID extraction pipeline (Phase B) |
| `scripts/watchdog.sh` | Bash watchdog for Phase A (nohup, auto-restart, reboot on done) |
| `scripts/watchdog.ps1` | PowerShell watchdog for Phase B (runs on login, auto-pulls GROBID) |
| `scripts/seed_data.py` | Build-time seeding (used for initial HF deployment, not currently active) |
| `scripts/download_pdfs.py` | Standalone download-only script (superseded by extract_fulltext.py) |
| `data/raw/fulltext_progress.json` | PyMuPDF progress tracker |
| `data/raw/pipeline_progress.json` | GROBID progress tracker |
| `data/raw/pdfs/` | Downloaded PDFs (~38 GB when complete) |
| `data/raw/tei/` | GROBID extracted text cache |
| `data/raw/extract_fulltext.log` | PyMuPDF pipeline log |
| `data/raw/pipeline.log` | GROBID pipeline log |
| `data/raw/watchdog.log` | Watchdog log |

### After pipeline completes
1. Re-chunk with fixed-size strategy (512 tokens) — currently abstract strategy (1 chunk/paper)
2. Re-embed ~557K chunks with BAAI/bge-base-en-v1.5
3. Re-run regex enrichment on full text
4. Upload new SQLite + ChromaDB to `thearkforyou/researchradar-data`
5. Redeploy HF Space (should stay within 50 GB disk / 16 GB RAM limits, estimated ~3.8 GB deployed data)

### Disk space estimates
- PDFs: ~38 GB (local only, not deployed)
- SQLite with full text: ~2.5 GB
- ChromaDB with 557K chunks: ~1.3 GB
- Total deployed: ~3.8 GB (within HF 50 GB limit)

## Frontend Notes

- API paths (fixed in `frontend/src/lib/api.ts`):
  - Trends: `/analytics/${type}/trend` (NOT `/analytics/trends/${type}`)
  - Top: `/analytics/${type}/top` (NOT `/analytics/top/${type}`)
  - Frontend `limit` param maps to backend `top_n` param
- SPA routing: FastAPI serves `index.html` for all non-API, non-static routes (configured in `app.py`)
- `deps.py`: calls `_db.create_schema()` on init to ensure tables exist on fresh deployments

## Local Machine Specs

- **CPU**: Intel Core i9-14900 (24 cores / 32 threads)
- **RAM**: 64 GB (typically ~38 GB free)
- **OS**: Windows (Git Bash for shell, Anaconda Python at `C:\Users\Admin\anaconda3\python.exe`)
- **Docker**: Docker Desktop with WSL2 backend, CLI at `C:\Program Files\Docker\Docker\resources\bin\docker.exe`
- **Note**: Git Bash may not be in PATH after reboot — use Python subprocess as fallback for shell commands

## Future TODOs

- **GROBID parallel acceleration**: Current run uses 1 worker + default GROBID threads. For future runs, use multi-worker + GROBID concurrency to leverage the 24-core i9:
  ```bash
  # Start GROBID with more threads and memory
  docker run -d --name grobid -p 8070:8070 --memory 16g --cpus 8 \
    -e GROBID_NB_THREADS=8 lfoppiano/grobid:0.8.1
  # Run pipeline with parallel workers
  python scripts/fulltext_pipeline.py --skip-download --workers 8
  ```
  Each GROBID engine instance uses ~500 MB RAM. With 64 GB available, can safely run 16 threads (8 GB for GROBID). Expected speedup: ~4-5x (12h → 2.5-3h). The `fulltext_pipeline.py` `grobid_extract_all()` function currently processes sequentially — would need a `ThreadPoolExecutor` to send concurrent requests.
- **Full-text enrichment**: Currently `EnrichmentPipeline._extract_text()` (pipeline.py:139) only uses title + abstract. Extend to support full paper text (from chunks table or full_text column) for higher-recall entity extraction. Key touchpoints: `_extract_text()`, `_get_unenriched_papers()` SQL query, and LLM prompt token budget (abstracts are ~200 tokens; full papers are ~5k-10k, may need chunked extraction or summarize-then-extract).
- **Entity-boosted retrieval**: Connect enrichment tags to the retrieval pipeline — extract entities from user queries, boost papers sharing those entities in ranking (see `src/retrieval/pipeline.py`).
- **Query expansion**: Use LLM to expand user queries with synonyms/related terms before retrieval.
- **Citation graph**: Parse references from GROBID TEI-XML to build a citation network for graph-based retrieval features.
