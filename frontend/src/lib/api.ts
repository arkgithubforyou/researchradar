/**
 * Typed API client for the ResearchRadar backend.
 * All 17 endpoints mapped to typed functions.
 */
import type {
  SearchRequest,
  SearchResponse,
  PaperBrowseParams,
  PaperListResponse,
  PaperDetail,
  TrendRequest,
  TrendResponse,
  RankedEntity,
  CooccurrenceRow,
  CooccurrenceType,
  EnrichmentStats,
  GrowthPoint,
  HealthResponse,
  EntityType,
} from "./types";

const BASE = "/api";

class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

async function request<T>(
  path: string,
  options?: RequestInit,
): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) {
    const body = await res.text().catch(() => "Unknown error");
    throw new ApiError(res.status, body);
  }
  return res.json() as Promise<T>;
}

function qs(params: Record<string, unknown>): string {
  const p = new URLSearchParams();
  for (const [k, v] of Object.entries(params)) {
    if (v != null && v !== "") p.set(k, String(v));
  }
  const s = p.toString();
  return s ? `?${s}` : "";
}

// ── Health ──────────────────────────────────────────────────────────

export function getHealth(): Promise<HealthResponse> {
  return request("/health");
}

// ── Search (RAG) ────────────────────────────────────────────────────

export function search(req: SearchRequest): Promise<SearchResponse> {
  return request("/search", {
    method: "POST",
    body: JSON.stringify(req),
  });
}

// ── Papers ──────────────────────────────────────────────────────────

/**
 * Resolve a composite venue value (e.g. "findings-acl") into
 * separate venue + volume query params for the API.
 */
function resolveVenueParams(
  params: PaperBrowseParams,
): Record<string, unknown> {
  const { venue, ...rest } = params;
  if (venue && venue.startsWith("findings-")) {
    return { ...rest, venue: "findings", volume: venue.slice("findings-".length) };
  }
  return { ...rest, venue };
}

export function getPapers(
  params: PaperBrowseParams = {},
): Promise<PaperListResponse> {
  return request(`/papers${qs(resolveVenueParams(params))}`);
}

export function getPaper(id: string): Promise<PaperDetail> {
  return request(`/papers/${encodeURIComponent(id)}`);
}

// ── Analytics ───────────────────────────────────────────────────────

export function getTrend(
  type: EntityType,
  req: TrendRequest,
): Promise<TrendResponse> {
  return request(`/analytics/${type}/trend`, {
    method: "POST",
    body: JSON.stringify(req),
  });
}

export function getTop(
  type: EntityType,
  params?: { year?: number; limit?: number },
): Promise<RankedEntity[]> {
  const { limit, ...rest } = params ?? {};
  const mapped = { ...rest, ...(limit != null ? { top_n: limit } : {}) };
  return request(`/analytics/${type}/top${qs(mapped)}`);
}

export function getCooccurrence(
  type: CooccurrenceType,
  params?: { limit?: number },
): Promise<CooccurrenceRow[]> {
  const { limit, ...rest } = params ?? {};
  const mapped = { ...rest, ...(limit != null ? { top_n: limit } : {}) };
  return request(`/analytics/cooccurrence/${type}${qs(mapped)}`);
}

export function getStats(): Promise<EnrichmentStats> {
  return request("/analytics/stats");
}

export function getGrowth(): Promise<GrowthPoint[]> {
  return request("/analytics/growth");
}

export function getVenues(): Promise<
  Array<{ venue: string; paper_count: number }>
> {
  return request("/analytics/venues");
}

export function getVenuesTotals(): Promise<
  Array<{ venue: string; paper_count: number }>
> {
  return request("/analytics/venues-total");
}
