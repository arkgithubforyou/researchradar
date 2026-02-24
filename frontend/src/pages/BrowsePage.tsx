import { useState, useCallback } from "react";
import { Link } from "react-router-dom";
import {
  BookOpen,
  ChevronLeft,
  ChevronRight,
  ExternalLink,
  Filter,
  Clock,
} from "lucide-react";
import { getPapers } from "@/lib/api";
import { useAsync, useDebounce } from "@/lib/hooks";
import { venueLabel, truncate, VENUES } from "@/lib/utils";
import Spinner from "@/components/Spinner";
import ErrorAlert from "@/components/ErrorAlert";
import EmptyState from "@/components/EmptyState";

const PAGE_SIZE = 20;

export default function BrowsePage() {
  const [venue, setVenue] = useState("");
  const [year, setYear] = useState("");
  const [method, setMethod] = useState("");
  const [dataset, setDataset] = useState("");
  const [author, setAuthor] = useState("");
  const [page, setPage] = useState(0);

  const debouncedMethod = useDebounce(method, 300);
  const debouncedDataset = useDebounce(dataset, 300);
  const debouncedAuthor = useDebounce(author, 300);

  const params = {
    venue: venue || null,
    year: year ? Number(year) : null,
    method: debouncedMethod || null,
    dataset: debouncedDataset || null,
    author: debouncedAuthor || null,
    limit: PAGE_SIZE,
    offset: page * PAGE_SIZE,
  };

  const { data, loading, error } = useAsync(
    () => getPapers(params),
    [venue, year, debouncedMethod, debouncedDataset, debouncedAuthor, page],
  );

  const handleFilterChange = useCallback(() => {
    setPage(0);
  }, []);

  const totalPages = data ? Math.ceil(data.count / PAGE_SIZE) : 0;
  const hasFilters = venue || year || method || dataset || author;

  return (
    <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      {/* ── Header ───────────────────────────────────────────────── */}
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Paper Browser</h1>
          <p className="text-sm text-gray-500 mt-1">
            {data
              ? `${data.count.toLocaleString()} papers found`
              : "Loading..."}
          </p>
        </div>
      </div>

      {/* ── Filters ──────────────────────────────────────────────── */}
      <div className="card p-4 mb-6">
        <div className="flex items-center gap-2 mb-3">
          <Filter className="w-4 h-4 text-gray-400" />
          <span className="text-sm font-medium text-gray-700">Filters</span>
        </div>
        <div className="flex flex-wrap items-center gap-3">
          <select
            value={venue}
            onChange={(e) => {
              setVenue(e.target.value);
              handleFilterChange();
            }}
            className="input !w-48 !py-2 text-sm"
          >
            <option value="">All venues</option>
            {VENUES.map((v) => (
              <option key={v.value} value={v.value}>
                {v.label}
              </option>
            ))}
          </select>

          <select
            value={year}
            onChange={(e) => {
              setYear(e.target.value);
              handleFilterChange();
            }}
            className="input !w-32 !py-2 text-sm"
          >
            <option value="">All years</option>
            {Array.from({ length: 6 }, (_, i) => 2025 - i).map((y) => (
              <option key={y} value={y}>
                {y}
              </option>
            ))}
          </select>

          <input
            type="text"
            value={method}
            onChange={(e) => {
              setMethod(e.target.value);
              handleFilterChange();
            }}
            placeholder="Method (e.g. BERT)"
            className="input !w-40 !py-2 text-sm"
          />

          <input
            type="text"
            value={dataset}
            onChange={(e) => {
              setDataset(e.target.value);
              handleFilterChange();
            }}
            placeholder="Dataset (e.g. SQuAD)"
            className="input !w-40 !py-2 text-sm"
          />

          <input
            type="text"
            value={author}
            onChange={(e) => {
              setAuthor(e.target.value);
              handleFilterChange();
            }}
            placeholder="Author (e.g. last name)"
            className="input !w-44 !py-2 text-sm"
          />

          {hasFilters && (
            <button
              onClick={() => {
                setVenue("");
                setYear("");
                setMethod("");
                setDataset("");
                setAuthor("");
                handleFilterChange();
              }}
              className="text-sm text-brand-600 hover:text-brand-700 font-medium"
            >
              Clear
            </button>
          )}
        </div>
      </div>

      {/* ── Error ────────────────────────────────────────────────── */}
      {error && <ErrorAlert message={error} />}

      {/* ── Loading ──────────────────────────────────────────────── */}
      {loading && (
        <div className="flex items-center justify-center py-20">
          <Spinner size="lg" />
        </div>
      )}

      {/* ── Empty ────────────────────────────────────────────────── */}
      {data && data.papers.length === 0 && !loading && (
        <EmptyState
          icon={<BookOpen className="w-12 h-12" />}
          title="No papers found"
          description="Try adjusting your filters or broadening your search."
        />
      )}

      {/* ── Paper list ───────────────────────────────────────────── */}
      {data && data.papers.length > 0 && !loading && (
        <>
          <div className="space-y-3">
            {data.papers.map((paper) => (
              <Link
                key={paper.id}
                to={`/paper/${encodeURIComponent(paper.id)}`}
                className="card-hover p-4 flex flex-col sm:flex-row sm:items-start gap-3 group block"
              >
                <div className="flex-1 min-w-0">
                  <h3 className="text-sm font-semibold text-gray-900 group-hover:text-brand-700 transition-colors line-clamp-2">
                    {paper.title}
                  </h3>
                  {paper.authors && paper.authors.length > 0 && (
                    <p className="text-xs text-gray-500 mt-1 line-clamp-1">
                      {paper.authors.join(", ")}
                    </p>
                  )}
                  {paper.abstract && (
                    <p className="text-xs text-gray-500 mt-1.5 line-clamp-2">
                      {truncate(paper.abstract, 250)}
                    </p>
                  )}
                  <div className="flex items-center gap-2 mt-2">
                    {paper.venue && (
                      <span className="badge-blue">{venueLabel(paper.venue)}</span>
                    )}
                    {paper.year && (
                      <span className="flex items-center gap-1 text-xs text-gray-400">
                        <Clock className="w-3 h-3" />
                        {paper.year}
                      </span>
                    )}
                  </div>
                </div>
                <ExternalLink className="w-4 h-4 text-gray-300 group-hover:text-brand-500 flex-shrink-0 mt-1 hidden sm:block" />
              </Link>
            ))}
          </div>

          {/* ── Pagination ─────────────────────────────────────────── */}
          {totalPages > 1 && (
            <div className="flex items-center justify-between mt-6 pt-4 border-t">
              <button
                onClick={() => setPage((p) => Math.max(0, p - 1))}
                disabled={page === 0}
                className="btn-secondary text-xs"
              >
                <ChevronLeft className="w-4 h-4" />
                Previous
              </button>
              <span className="text-sm text-gray-500">
                Page {page + 1} of {totalPages}
              </span>
              <button
                onClick={() => setPage((p) => Math.min(totalPages - 1, p + 1))}
                disabled={page >= totalPages - 1}
                className="btn-secondary text-xs"
              >
                Next
                <ChevronRight className="w-4 h-4" />
              </button>
            </div>
          )}
        </>
      )}
    </div>
  );
}
