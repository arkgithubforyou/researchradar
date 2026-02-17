import {
  TrendingUp,
  FileText,
  Beaker,
  Database,
  Tag,
  BookOpen,
} from "lucide-react";
import { getGrowth, getTop, getVenues, getStats } from "@/lib/api";
import { useAsync } from "@/lib/hooks";
import { formatCount } from "@/lib/utils";
import StatCard from "@/components/StatCard";
import Spinner from "@/components/Spinner";
import GrowthChart from "@/components/GrowthChart";
import TopEntitiesChart from "@/components/TopEntitiesChart";
import VenueChart from "@/components/VenueChart";
import TrendExplorer from "@/components/TrendExplorer";
import CooccurrenceTable from "@/components/CooccurrenceTable";

export default function DashboardPage() {
  const { data: stats, loading: statsLoading } = useAsync(() => getStats(), []);
  const { data: growth, loading: growthLoading } = useAsync(() => getGrowth(), []);
  const { data: venues, loading: venuesLoading } = useAsync(() => getVenues(), []);
  const { data: topMethods, loading: methodsLoading } = useAsync(
    () => getTop("methods", { limit: 10 }),
    [],
  );
  const { data: topDatasets, loading: datasetsLoading } = useAsync(
    () => getTop("datasets", { limit: 10 }),
    [],
  );

  const isLoading = statsLoading || growthLoading;

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      {/* ── Header ───────────────────────────────────────────────── */}
      <div className="mb-8">
        <h1 className="text-2xl font-bold text-gray-900">Dashboard</h1>
        <p className="text-sm text-gray-500 mt-1">
          Trends and analytics across the paper corpus
        </p>
      </div>

      {/* ── Stats grid ───────────────────────────────────────────── */}
      {isLoading ? (
        <div className="flex items-center justify-center py-12">
          <Spinner size="lg" />
        </div>
      ) : (
        stats && (
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4 mb-8">
            <StatCard
              label="Total Papers"
              value={formatCount(stats.total_papers)}
              icon={<FileText className="w-5 h-5" />}
            />
            <StatCard
              label="Methods"
              value={formatCount(stats.total_methods)}
              icon={<Beaker className="w-5 h-5" />}
              subtitle={`${stats.papers_with_methods} papers enriched`}
            />
            <StatCard
              label="Datasets"
              value={formatCount(stats.total_datasets)}
              icon={<Database className="w-5 h-5" />}
            />
            <StatCard
              label="Tasks"
              value={formatCount(stats.total_tasks)}
              icon={<Tag className="w-5 h-5" />}
            />
          </div>
        )
      )}

      {/* ── Growth chart + Venues ─────────────────────────────────── */}
      <div className="grid gap-6 lg:grid-cols-3 mb-8">
        <div className="lg:col-span-2 card p-5">
          <div className="flex items-center gap-2 mb-4">
            <TrendingUp className="w-4 h-4 text-gray-400" />
            <h2 className="text-sm font-semibold text-gray-900">
              Papers Per Year
            </h2>
          </div>
          {growthLoading ? (
            <div className="h-64 flex items-center justify-center">
              <Spinner />
            </div>
          ) : growth ? (
            <GrowthChart data={growth} />
          ) : null}
        </div>

        <div className="card p-5">
          <div className="flex items-center gap-2 mb-4">
            <BookOpen className="w-4 h-4 text-gray-400" />
            <h2 className="text-sm font-semibold text-gray-900">By Venue</h2>
          </div>
          {venuesLoading ? (
            <div className="h-64 flex items-center justify-center">
              <Spinner />
            </div>
          ) : venues ? (
            <VenueChart data={venues} />
          ) : null}
        </div>
      </div>

      {/* ── Top entities ─────────────────────────────────────────── */}
      <div className="grid gap-6 lg:grid-cols-2 mb-8">
        <div className="card p-5">
          <div className="flex items-center gap-2 mb-4">
            <Beaker className="w-4 h-4 text-purple-500" />
            <h2 className="text-sm font-semibold text-gray-900">
              Top Methods
            </h2>
          </div>
          {methodsLoading ? (
            <div className="h-64 flex items-center justify-center">
              <Spinner />
            </div>
          ) : topMethods ? (
            <TopEntitiesChart data={topMethods} color="#ae3ec9" />
          ) : null}
        </div>

        <div className="card p-5">
          <div className="flex items-center gap-2 mb-4">
            <Database className="w-4 h-4 text-emerald-500" />
            <h2 className="text-sm font-semibold text-gray-900">
              Top Datasets
            </h2>
          </div>
          {datasetsLoading ? (
            <div className="h-64 flex items-center justify-center">
              <Spinner />
            </div>
          ) : topDatasets ? (
            <TopEntitiesChart data={topDatasets} color="#37b24d" />
          ) : null}
        </div>
      </div>

      {/* ── Trend explorer ───────────────────────────────────────── */}
      <div className="card p-5 mb-8">
        <TrendExplorer />
      </div>

      {/* ── Co-occurrence ────────────────────────────────────────── */}
      <div className="card p-5">
        <CooccurrenceTable />
      </div>
    </div>
  );
}
