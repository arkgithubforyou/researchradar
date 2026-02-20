import { useState, useCallback, useRef, useEffect } from "react";
import { Link } from "react-router-dom";
import {
  Search,
  Sparkles,
  ExternalLink,
  ChevronDown,
  ChevronUp,
  FileText,
  Clock,
  Zap,
} from "lucide-react";
import { search } from "@/lib/api";
import { useLazyAsync } from "@/lib/hooks";
import { cn, venueLabel, VENUES } from "@/lib/utils";
import type { SearchRequest } from "@/lib/types";
import Spinner from "@/components/Spinner";
import ErrorAlert from "@/components/ErrorAlert";

const EXAMPLE_QUERIES = [
  "What are the latest advances in retrieval-augmented generation?",
  "How do chain-of-thought prompting techniques compare?",
  "Which datasets are used to evaluate multilingual models?",
  "What methods improve low-resource NER performance?",
];

export default function SearchPage() {
  const [query, setQuery] = useState("");
  const [showFilters, setShowFilters] = useState(false);
  const [yearMin, setYearMin] = useState<string>("");
  const [yearMax, setYearMax] = useState<string>("");
  const [venue, setVenue] = useState<string>("");
  const inputRef = useRef<HTMLInputElement>(null);

  const { data, loading, error, execute, reset } = useLazyAsync(search);

  const handleSearch = useCallback(
    (q?: string) => {
      const searchQuery = q ?? query;
      if (!searchQuery.trim()) return;
      if (q) setQuery(q);

      const req: SearchRequest = {
        query: searchQuery.trim(),
        top_k: 5,
        year_min: yearMin ? Number(yearMin) : null,
        year_max: yearMax ? Number(yearMax) : null,
        venue: venue || null,
      };
      execute(req);
    },
    [query, yearMin, yearMax, venue, execute],
  );

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSearch();
    }
  };

  // Focus input on mount
  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  const hasResult = data !== null;

  return (
    <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
      {/* ── Hero section ─────────────────────────────────────────── */}
      {!hasResult && !loading && (
        <div className="pt-20 pb-8 text-center">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-brand-50 text-brand-700 text-xs font-medium mb-6">
            <Sparkles className="w-3.5 h-3.5" />
            RAG-powered research exploration
          </div>
          <h1 className="text-4xl sm:text-5xl font-bold text-gray-900 tracking-tight">
            Explore NLP research
          </h1>
          <p className="mt-4 text-lg text-gray-500 max-w-2xl mx-auto">
            Ask questions about NLP and ML papers. Get AI-generated answers
            grounded in real publications from ACL, EMNLP, and NAACL.
          </p>
        </div>
      )}

      {/* ── Search box ───────────────────────────────────────────── */}
      <div className={cn("w-full", hasResult || loading ? "pt-6 pb-4" : "pt-4 pb-6")}>
        <div className="card p-2 flex items-center gap-2 shadow-sm hover:shadow-md transition-shadow">
          <Search className="w-5 h-5 text-gray-400 ml-3 flex-shrink-0" />
          <input
            ref={inputRef}
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask a research question..."
            className="flex-1 py-2.5 px-2 text-sm bg-transparent border-none outline-none placeholder:text-gray-400"
          />
          <button
            onClick={() => setShowFilters(!showFilters)}
            className={cn(
              "btn-ghost !px-2 !py-1.5 text-xs",
              showFilters && "bg-gray-100",
            )}
          >
            Filters
            {showFilters ? (
              <ChevronUp className="w-3.5 h-3.5" />
            ) : (
              <ChevronDown className="w-3.5 h-3.5" />
            )}
          </button>
          <button
            onClick={() => handleSearch()}
            disabled={!query.trim() || loading}
            className="btn-primary !py-2"
          >
            {loading ? <Spinner size="sm" className="text-white" /> : "Search"}
          </button>
        </div>

        {/* Filters row */}
        {showFilters && (
          <div className="mt-3 flex flex-wrap items-center gap-3 px-1">
            <div className="flex items-center gap-2">
              <label className="text-xs text-gray-500 font-medium">Year</label>
              <input
                type="number"
                value={yearMin}
                onChange={(e) => setYearMin(e.target.value)}
                placeholder="From"
                className="input !w-24 !py-1.5 text-xs"
              />
              <span className="text-gray-300">&ndash;</span>
              <input
                type="number"
                value={yearMax}
                onChange={(e) => setYearMax(e.target.value)}
                placeholder="To"
                className="input !w-24 !py-1.5 text-xs"
              />
            </div>
            <div className="flex items-center gap-2">
              <label className="text-xs text-gray-500 font-medium">Venue</label>
              <select
                value={venue}
                onChange={(e) => setVenue(e.target.value)}
                className="input !w-44 !py-1.5 text-xs"
              >
                <option value="">All</option>
                {VENUES.map((v) => (
                  <option key={v.value} value={v.value}>
                    {v.label}
                  </option>
                ))}
              </select>
            </div>
            {(yearMin || yearMax || venue) && (
              <button
                onClick={() => {
                  setYearMin("");
                  setYearMax("");
                  setVenue("");
                }}
                className="text-xs text-brand-600 hover:text-brand-700 font-medium"
              >
                Clear filters
              </button>
            )}
          </div>
        )}
      </div>

      {/* ── Example queries (only when no result) ───────────────── */}
      {!hasResult && !loading && !error && (
        <div className="pb-8">
          <p className="text-xs font-medium text-gray-400 uppercase tracking-wider mb-3 px-1">
            Try asking
          </p>
          <div className="grid gap-2 sm:grid-cols-2">
            {EXAMPLE_QUERIES.map((q) => (
              <button
                key={q}
                onClick={() => handleSearch(q)}
                className="text-left card-hover p-3 flex items-start gap-2.5 group"
              >
                <Zap className="w-4 h-4 text-amber-400 mt-0.5 flex-shrink-0" />
                <span className="text-sm text-gray-600 group-hover:text-gray-900 transition-colors">
                  {q}
                </span>
              </button>
            ))}
          </div>
        </div>
      )}

      {/* ── Loading skeleton ─────────────────────────────────────── */}
      {loading && (
        <div className="space-y-4 pb-12">
          <div className="card p-6">
            <div className="flex items-center gap-2 mb-4">
              <Spinner size="sm" />
              <span className="text-sm text-gray-500">
                Searching and generating answer...
              </span>
            </div>
            <div className="space-y-3">
              <div className="shimmer h-4 rounded w-full" />
              <div className="shimmer h-4 rounded w-11/12" />
              <div className="shimmer h-4 rounded w-4/5" />
              <div className="shimmer h-4 rounded w-9/12" />
            </div>
          </div>
          <div className="grid gap-3 sm:grid-cols-2">
            {[1, 2, 3, 4].map((i) => (
              <div key={i} className="card p-4">
                <div className="shimmer h-4 rounded w-3/4 mb-2" />
                <div className="shimmer h-3 rounded w-1/2" />
              </div>
            ))}
          </div>
        </div>
      )}

      {/* ── Error ────────────────────────────────────────────────── */}
      {error && !loading && (
        <div className="pb-8">
          <ErrorAlert message={error} onRetry={() => handleSearch()} />
        </div>
      )}

      {/* ── Answer ───────────────────────────────────────────────── */}
      {data && !loading && (
        <div className="space-y-5 pb-12">
          {/* Answer card */}
          <div className="card p-6">
            <div className="flex items-center gap-2 mb-4">
              <div className="w-6 h-6 rounded-full bg-brand-100 flex items-center justify-center">
                <Sparkles className="w-3.5 h-3.5 text-brand-600" />
              </div>
              <span className="text-sm font-medium text-gray-900">
                AI Answer
              </span>
              <span className="badge-gray ml-auto">
                {data.model}
              </span>
            </div>
            <div className="prose prose-sm prose-gray max-w-none text-gray-700 leading-relaxed whitespace-pre-wrap">
              {data.answer}
            </div>
          </div>

          {/* Sources */}
          {data.sources.length > 0 && (
            <div>
              <div className="flex items-center gap-2 mb-3 px-1">
                <FileText className="w-4 h-4 text-gray-400" />
                <h3 className="text-sm font-medium text-gray-700">
                  Sources ({data.sources.length} papers)
                </h3>
              </div>
              <div className="grid gap-3 sm:grid-cols-2">
                {data.sources.map((src, i) => (
                  <Link
                    key={`${src.paper_id}-${i}`}
                    to={`/paper/${encodeURIComponent(src.paper_id)}`}
                    className="card-hover p-4 group"
                  >
                    <div className="flex items-start justify-between gap-2">
                      <h4 className="text-sm font-medium text-gray-900 group-hover:text-brand-700 transition-colors line-clamp-2">
                        {src.title}
                      </h4>
                      <ExternalLink className="w-3.5 h-3.5 text-gray-300 group-hover:text-brand-500 flex-shrink-0 mt-0.5" />
                    </div>
                    <div className="flex items-center gap-2 mt-2">
                      <span className="badge-blue">{venueLabel(src.venue)}</span>
                      <span className="flex items-center gap-1 text-xs text-gray-400">
                        <Clock className="w-3 h-3" />
                        {src.year}
                      </span>
                    </div>
                  </Link>
                ))}
              </div>
            </div>
          )}

          {/* New search prompt */}
          <div className="text-center pt-4">
            <button
              onClick={() => {
                reset();
                setQuery("");
                inputRef.current?.focus();
              }}
              className="text-sm text-brand-600 hover:text-brand-700 font-medium"
            >
              Ask another question
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
