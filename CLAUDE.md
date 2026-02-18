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

## API

`uvicorn src.api.app:app` — 17 endpoints under `/api`:

| Endpoint | Purpose |
|---|---|
| `GET /api/health` | Health + paper/chunk counts |
| `POST /api/search` | RAG search with year/venue filters |
| `GET /api/papers` | Browse with filters |
| `GET /api/papers/{id}` | Paper detail (`:path` for `::` IDs) |
| `POST /api/analytics/trends/{type}` | Trend by entity (methods/datasets/tasks/topics) |
| `GET /api/analytics/top/{type}` | Top-N ranking (methods/datasets/tasks/topics) |
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

## Infrastructure (Terraform)

Three deployment options:

### Hugging Face Spaces — Free ($0/month, easiest)

`Dockerfile.hf` + `scripts/deploy_hf.sh` — Docker Space on HF free tier (2 vCPU, 16 GB RAM).

```bash
# One-time: create Space at huggingface.co/new-space (Docker SDK, CPU basic)
# Set GROQ_API_KEY as a Secret in Space Settings
huggingface-cli login
./scripts/deploy_hf.sh <username>/researchradar
```

- Sleeps after 48h inactivity; `.github/workflows/keep-alive.yml` pings every 12h to prevent this
- Set `HF_SPACE_URL` as a GitHub repo variable for the keep-alive to work
- Redeployments: re-run `deploy_hf.sh` or push to the HF Space repo directly

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

## Future TODOs

- **Full-text enrichment**: Currently `EnrichmentPipeline._extract_text()` (pipeline.py:139) only uses title + abstract. Extend to support full paper text (from chunks table or full_text column) for higher-recall entity extraction. Key touchpoints: `_extract_text()`, `_get_unenriched_papers()` SQL query, and LLM prompt token budget (abstracts are ~200 tokens; full papers are ~5k-10k, may need chunked extraction or summarize-then-extract).
- **Entity-boosted retrieval**: Connect enrichment tags to the retrieval pipeline — extract entities from user queries, boost papers sharing those entities in ranking (see `src/retrieval/pipeline.py`).
