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
- `frontend/` — React SPA: `src/pages/` (Search, Browse, Paper, Dashboard), `src/components/` (Layout, StatCard, EntityListModal, charts, tables), `src/lib/` (api.ts, types.ts, hooks.ts)
- `src/evaluation/` — `metrics.py` (P@K, R@K, MRR, NDCG, ROUGE-L), `dataset.py` (eval dataset loader), `runner.py` (eval runner), `ablation.py` (ablation study framework)
- `scripts/` — `ingest.py`, `enrich.py`, `smoke_test_llm.py`, `write_questions.py`, `annotate.py`, `annotate_generation.py`, `retrieval_eval.py`, `run_ablation.py`, `embed_chunks.py`, `extract_fulltext.py`, `fulltext_pipeline.py`, `build_bm25_index.py`
- `data/` — `questions.json`, `eval_set.json`, `researchradar.db`, `chroma_db/`, `bm25_index.pkl`, `raw/`

## Progress

- [x] Phase 1: Loaders, SQLite, chunking, ChromaDB, embeddings, S3, ingest CLI
- [x] Phase 2: BM25, vector, hybrid RRF, cross-encoder reranker, retrieval pipeline
- [x] Phase 3: LLM backend ABC, Ollama + Groq backends, RAG engine, smoke test
- [x] Phase 4: LLM entity extraction, regex fallback, SQL analytics
- [x] Phase 5: FastAPI endpoints (17 routes), eval framework, annotation pipeline, ablation study
- [x] Phase 6: Docker Compose, deploy, CI/CD, Terraform
- [x] Phase 7: React frontend (Search, Browse, Paper Detail, Dashboard), static serving, rate limiting, HTTPS Terraform
- [ ] Phase 8: Full-text extraction (IN PROGRESS — see DEV_README.md)

## API

`uvicorn src.api.app:app` — 18 endpoints under `/api`:

| Endpoint | Purpose |
|---|---|
| `GET /api/health` | Health + paper/chunk counts |
| `POST /api/search` | RAG search with year/venue filters |
| `GET /api/papers` | Browse with filters |
| `GET /api/papers/{id}` | Paper detail (`:path` for `::` IDs) |
| `POST /api/analytics/{type}/trend` | Trend by entity (methods/datasets/tasks/topics) |
| `GET /api/analytics/{type}/top` | Top-N ranking (methods/datasets/tasks/topics) |
| `GET /api/analytics/{type}/list` | All unique entities with paper counts |
| `GET /api/analytics/cooccurrence/{type}` | Co-occurrence (method-dataset, method-task) |
| `GET /api/analytics/stats` | Enrichment stats (unique counts) |
| `GET /api/analytics/growth` | Papers per year |
| `GET /api/analytics/venues` | Papers per venue |

DI via module singletons in `deps.py`. Lifespan loads models once at startup; `is_initialized()` guard skips loading during tests.

## Deployment

**Live at**: https://thearkforyou-researchradar.hf.space (HF Spaces free tier)

**Currently deployed**: 26,544 papers, 394,777 chunks. Data: SQLite 2.39 GB + ChromaDB 8.56 GB + BM25 index 866 MB = ~11.8 GB total.

**Pre-serialized BM25**: The BM25 index is pre-built locally (`python -m scripts.build_bm25_index`), uploaded to the HF dataset repo as `bm25_index.pkl`, and downloaded at Docker build time. Startup loads the pickle (~1s) instead of tokenizing 394K chunks (~16 min). Falls back to live rebuild if the file is missing.

HF free tier limits: 2 vCPU, 16 GB RAM, 50 GB disk, ~30 min Docker build timeout. See DEV_README.md for full resource analysis and scaling projections.

## HF Credentials

- **Token**: (stored in HF Spaces secrets and local env — do NOT commit)
- **Username**: `thearkforyou`
- **Space**: `thearkforyou/researchradar`
- **Dataset repo**: `thearkforyou/researchradar-data` (holds pre-built SQLite + ChromaDB)
- **GitHub**: `arkgithubforyou/researchradar`

## Full-Text Extraction (ONGOING)

Two-phase pipeline running unattended (PyMuPDF → GROBID). See DEV_README.md for full architecture.

**After pipeline completes**:
1. Re-chunk with fixed-size strategy (512 tokens)
2. Re-embed chunks with BAAI/bge-base-en-v1.5
3. Re-run regex enrichment on full text
4. Upload new SQLite + ChromaDB to HF dataset repo
5. Redeploy HF Space

**Disk estimates** (post full-text): SQLite ~2.5 GB, ChromaDB ~12 GB (with 557K chunks), total deployed ~14.5 GB.

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

- **Switch ChromaDB → FAISS**: ChromaDB 0.4.24 stores embeddings redundantly in both SQLite and HNSW (8.56 GB total, only 1.2 GB needed). FAISS would cut storage to ~1.2 GB and unblock dataset scaling — build timeout (~30 min) is the current bottleneck. See DEV_README.md for full analysis.
- **GROBID parallel acceleration**: Use multi-worker + GROBID concurrency (`--workers 8`, `GROBID_NB_THREADS=8`). Expected 4-5x speedup. Needs `ThreadPoolExecutor` in `fulltext_pipeline.py:grobid_extract_all()`.
- **Full-text enrichment**: `EnrichmentPipeline._extract_text()` (pipeline.py:139) only uses abstract. Extend to full text with chunked extraction or summarize-then-extract.
- **Entity-boosted retrieval**: Extract entities from queries, boost matching papers in ranking.
- **Query expansion**: LLM-powered query expansion with synonyms/related terms.
- **Citation graph**: Parse GROBID TEI-XML references for graph-based retrieval.
