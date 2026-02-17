import { useState, useCallback } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from "recharts";
import { TrendingUp, Plus, X } from "lucide-react";
import { getTrend } from "@/lib/api";
import { CHART_COLORS } from "@/lib/utils";
import type { EntityType, TrendResponse } from "@/lib/types";
import Spinner from "./Spinner";

const ENTITY_OPTIONS: Array<{ value: EntityType; label: string }> = [
  { value: "methods", label: "Method" },
  { value: "datasets", label: "Dataset" },
  { value: "tasks", label: "Task" },
  { value: "topics", label: "Topic" },
];

const SUGGESTIONS = [
  { type: "methods" as EntityType, name: "Transformer" },
  { type: "methods" as EntityType, name: "BERT" },
  { type: "datasets" as EntityType, name: "SQuAD" },
  { type: "methods" as EntityType, name: "GPT" },
  { type: "tasks" as EntityType, name: "Machine Translation" },
];

interface TrendLine {
  type: EntityType;
  name: string;
  data: TrendResponse;
}

export default function TrendExplorer() {
  const [lines, setLines] = useState<TrendLine[]>([]);
  const [entityType, setEntityType] = useState<EntityType>("methods");
  const [entityName, setEntityName] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const addTrend = useCallback(
    async (type?: EntityType, name?: string) => {
      const t = type ?? entityType;
      const n = (name ?? entityName).trim();
      if (!n) return;

      // Don't add duplicate
      if (lines.some((l) => l.type === t && l.name.toLowerCase() === n.toLowerCase())) {
        return;
      }

      setLoading(true);
      setError(null);
      try {
        const data = await getTrend(t, { name: n });
        setLines((prev) => [...prev, { type: t, name: n, data }]);
        setEntityName("");
      } catch (err: unknown) {
        setError(err instanceof Error ? err.message : "Failed to load trend");
      } finally {
        setLoading(false);
      }
    },
    [entityType, entityName, lines],
  );

  const removeTrend = (index: number) => {
    setLines((prev) => prev.filter((_, i) => i !== index));
  };

  // Merge all lines into a single dataset keyed by year
  const mergedData = (() => {
    const byYear = new Map<number, Record<string, number>>();
    for (const line of lines) {
      for (const pt of line.data.trend) {
        const row = byYear.get(pt.year) ?? { year: pt.year };
        row[line.name] = pt.paper_count;
        byYear.set(pt.year, row);
      }
    }
    return Array.from(byYear.values()).sort(
      (a, b) => (a["year"] as number) - (b["year"] as number),
    );
  })();

  return (
    <div>
      <div className="flex items-center gap-2 mb-4">
        <TrendingUp className="w-4 h-4 text-gray-400" />
        <h2 className="text-sm font-semibold text-gray-900">
          Trend Explorer
        </h2>
      </div>

      {/* ── Input row ────────────────────────────────────────────── */}
      <div className="flex flex-wrap items-center gap-2 mb-4">
        <select
          value={entityType}
          onChange={(e) => setEntityType(e.target.value as EntityType)}
          className="input !w-32 !py-2 text-sm"
        >
          {ENTITY_OPTIONS.map((o) => (
            <option key={o.value} value={o.value}>
              {o.label}
            </option>
          ))}
        </select>
        <input
          type="text"
          value={entityName}
          onChange={(e) => setEntityName(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && addTrend()}
          placeholder="e.g. Transformer, BERT, SQuAD..."
          className="input !w-64 !py-2 text-sm"
        />
        <button
          onClick={() => addTrend()}
          disabled={!entityName.trim() || loading}
          className="btn-primary !py-2"
        >
          {loading ? <Spinner size="sm" className="text-white" /> : <Plus className="w-4 h-4" />}
          Add
        </button>
      </div>

      {/* Suggestions */}
      {lines.length === 0 && (
        <div className="flex flex-wrap gap-2 mb-4">
          <span className="text-xs text-gray-400">Quick add:</span>
          {SUGGESTIONS.map((s) => (
            <button
              key={`${s.type}-${s.name}`}
              onClick={() => addTrend(s.type, s.name)}
              className="badge-gray hover:bg-gray-200 cursor-pointer transition-colors"
            >
              {s.name}
            </button>
          ))}
        </div>
      )}

      {error && <p className="text-sm text-red-500 mb-3">{error}</p>}

      {/* Active trend pills */}
      {lines.length > 0 && (
        <div className="flex flex-wrap gap-2 mb-4">
          {lines.map((line, i) => (
            <span
              key={`${line.type}-${line.name}`}
              className="inline-flex items-center gap-1 px-2.5 py-1 rounded-full text-xs font-medium text-white"
              style={{ backgroundColor: CHART_COLORS[i % CHART_COLORS.length] }}
            >
              {line.name}
              <button
                onClick={() => removeTrend(i)}
                className="ml-0.5 hover:opacity-70"
              >
                <X className="w-3 h-3" />
              </button>
            </span>
          ))}
        </div>
      )}

      {/* Chart */}
      {mergedData.length > 0 ? (
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={mergedData} margin={{ top: 4, right: 12, left: -20, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#f1f3f5" />
            <XAxis
              dataKey="year"
              tick={{ fontSize: 12, fill: "#868e96" }}
              axisLine={false}
              tickLine={false}
            />
            <YAxis
              tick={{ fontSize: 12, fill: "#868e96" }}
              axisLine={false}
              tickLine={false}
            />
            <Tooltip
              contentStyle={{
                fontSize: 13,
                borderRadius: 8,
                border: "1px solid #e9ecef",
                boxShadow: "0 4px 12px rgba(0,0,0,0.08)",
              }}
            />
            <Legend iconType="circle" iconSize={8} />
            {lines.map((line, i) => (
              <Line
                key={`${line.type}-${line.name}`}
                type="monotone"
                dataKey={line.name}
                stroke={CHART_COLORS[i % CHART_COLORS.length]}
                strokeWidth={2}
                dot={{ r: 3, strokeWidth: 0 }}
                activeDot={{ r: 5, strokeWidth: 2, stroke: "#fff" }}
                connectNulls
              />
            ))}
          </LineChart>
        </ResponsiveContainer>
      ) : (
        <div className="h-48 flex items-center justify-center text-sm text-gray-400">
          Add entities above to compare their trends over time
        </div>
      )}
    </div>
  );
}
