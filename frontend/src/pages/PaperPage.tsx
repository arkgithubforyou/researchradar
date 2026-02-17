import { useParams, Link } from "react-router-dom";
import {
  ArrowLeft,
  ExternalLink,
  Users,
  Beaker,
  Database,
  Calendar,
  MapPin,
} from "lucide-react";
import { getPaper } from "@/lib/api";
import { useAsync } from "@/lib/hooks";
import { venueLabel } from "@/lib/utils";
import Spinner from "@/components/Spinner";
import ErrorAlert from "@/components/ErrorAlert";

export default function PaperPage() {
  const { paperId } = useParams<{ paperId: string }>();
  const decodedId = decodeURIComponent(paperId ?? "");

  const { data: paper, loading, error } = useAsync(
    () => getPaper(decodedId),
    [decodedId],
  );

  if (loading) {
    return (
      <div className="flex items-center justify-center py-32">
        <Spinner size="lg" />
      </div>
    );
  }

  if (error) {
    return (
      <div className="max-w-3xl mx-auto px-4 py-8">
        <ErrorAlert message={error} />
      </div>
    );
  }

  if (!paper) return null;

  return (
    <div className="max-w-3xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      {/* Back link */}
      <Link
        to="/browse"
        className="inline-flex items-center gap-1.5 text-sm text-gray-500 hover:text-gray-900 mb-6 transition-colors"
      >
        <ArrowLeft className="w-4 h-4" />
        Back to papers
      </Link>

      {/* ── Paper header ─────────────────────────────────────────── */}
      <article>
        <div className="flex items-center gap-2 mb-3">
          {paper.venue && (
            <span className="badge-blue">{venueLabel(paper.venue)}</span>
          )}
          {paper.year && (
            <span className="flex items-center gap-1 text-xs text-gray-400">
              <Calendar className="w-3 h-3" />
              {paper.year}
            </span>
          )}
        </div>

        <h1 className="text-2xl font-bold text-gray-900 leading-tight">
          {paper.title}
        </h1>

        {/* Authors */}
        {paper.authors.length > 0 && (
          <div className="flex items-start gap-2 mt-4">
            <Users className="w-4 h-4 text-gray-400 mt-0.5 flex-shrink-0" />
            <p className="text-sm text-gray-600">
              {paper.authors.join(", ")}
            </p>
          </div>
        )}

        {/* External link */}
        {paper.url && (
          <a
            href={paper.url}
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-1.5 text-sm text-brand-600 hover:text-brand-700 mt-3 font-medium"
          >
            <ExternalLink className="w-4 h-4" />
            View original paper
          </a>
        )}

        {/* ── Abstract ─────────────────────────────────────────────── */}
        {paper.abstract && (
          <div className="mt-8">
            <h2 className="text-sm font-semibold text-gray-900 uppercase tracking-wider mb-3">
              Abstract
            </h2>
            <div className="card p-5">
              <p className="text-sm text-gray-700 leading-relaxed whitespace-pre-wrap">
                {paper.abstract}
              </p>
            </div>
          </div>
        )}

        {/* ── Enrichment tags ──────────────────────────────────────── */}
        <div className="mt-8 grid gap-6 sm:grid-cols-2">
          {/* Methods */}
          {paper.methods.length > 0 && (
            <div>
              <div className="flex items-center gap-2 mb-3">
                <Beaker className="w-4 h-4 text-purple-500" />
                <h2 className="text-sm font-semibold text-gray-900">
                  Methods
                </h2>
              </div>
              <div className="flex flex-wrap gap-2">
                {paper.methods.map((m, i) => (
                  <span key={i} className="badge-purple">
                    {m.name}
                  </span>
                ))}
              </div>
            </div>
          )}

          {/* Datasets */}
          {paper.datasets.length > 0 && (
            <div>
              <div className="flex items-center gap-2 mb-3">
                <Database className="w-4 h-4 text-emerald-500" />
                <h2 className="text-sm font-semibold text-gray-900">
                  Datasets
                </h2>
              </div>
              <div className="flex flex-wrap gap-2">
                {paper.datasets.map((d, i) => (
                  <span key={i} className="badge-green">
                    {d.name}
                  </span>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* ── Paper ID (subtle footer) ─────────────────────────────── */}
        <div className="mt-10 pt-4 border-t">
          <div className="flex items-center gap-2">
            <MapPin className="w-3 h-3 text-gray-300" />
            <code className="text-xs text-gray-400 font-mono">{paper.id}</code>
          </div>
        </div>
      </article>
    </div>
  );
}
