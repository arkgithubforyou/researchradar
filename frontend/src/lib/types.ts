// ── Request types ──────────────────────────────────────────────────

export interface SearchRequest {
  query: string;
  top_k?: number;
  year_min?: number | null;
  year_max?: number | null;
  venue?: string | null;
}

export interface PaperBrowseParams {
  venue?: string | null;
  volume?: string | null;
  year?: number | null;
  method?: string | null;
  dataset?: string | null;
  author?: string | null;
  limit?: number;
  offset?: number;
}

export interface TrendRequest {
  name: string;
}

// ── Response types ─────────────────────────────────────────────────

export interface SourcePaper {
  paper_id: string;
  title: string;
  year: number;
  venue: string | null;
  chunk_type: string;
  authors?: string[];
  used_in_answer?: boolean;
}

export interface SearchResponse {
  answer: string;
  sources: SourcePaper[];
  model: string;
  usage: Record<string, unknown>;
}

export interface PaperSummary {
  id: string;
  title: string;
  abstract: string | null;
  year: number | null;
  venue: string | null;
  url: string | null;
  authors?: string[];
}

export interface PaperDetail {
  id: string;
  title: string;
  abstract: string | null;
  year: number | null;
  venue: string | null;
  url: string | null;
  authors: string[];
  methods: Array<{ name: string; [k: string]: unknown }>;
  datasets: Array<{ name: string; [k: string]: unknown }>;
}

export interface PaperListResponse {
  papers: PaperSummary[];
  count: number;
  limit: number;
  offset: number;
}

export interface TrendPoint {
  year: number;
  paper_count: number;
}

export interface TrendResponse {
  name: string;
  trend: TrendPoint[];
}

export interface RankedEntity {
  name: string;
  year: number;
  count: number;
  rank: number;
}

export interface CooccurrenceRow {
  entity_a: string;
  entity_b: string;
  co_count: number;
}

export interface EnrichmentStats {
  total_papers: number;
  total_methods: number;
  total_datasets: number;
  total_tasks: number;
  total_topics: number;
  papers_with_methods: number;
}

export interface GrowthPoint {
  year: number;
  paper_count: number;
  prev_count: number | null;
  growth_pct: number | null;
}

export interface HealthResponse {
  status: string;
  paper_count: number;
  chunk_count: number;
}

export interface EntityListItem {
  name: string;
  count: number;
}

// Entity type for analytics endpoints
export type EntityType = "methods" | "datasets" | "tasks" | "topics";
export type CooccurrenceType = "method-dataset" | "method-task";
