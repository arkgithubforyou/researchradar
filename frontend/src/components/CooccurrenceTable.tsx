import { useState } from "react";
import { Layers, ArrowRight } from "lucide-react";
import { getCooccurrence } from "@/lib/api";
import { useAsync } from "@/lib/hooks";
import type { CooccurrenceType } from "@/lib/types";
import Spinner from "./Spinner";

const TYPE_OPTIONS: Array<{ value: CooccurrenceType; label: string }> = [
  { value: "method-dataset", label: "Method + Dataset" },
  { value: "method-task", label: "Method + Task" },
];

export default function CooccurrenceTable() {
  const [type, setType] = useState<CooccurrenceType>("method-dataset");

  const { data, loading, error } = useAsync(
    () => getCooccurrence(type, { limit: 15 }),
    [type],
  );

  return (
    <div>
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <Layers className="w-4 h-4 text-gray-400" />
          <h2 className="text-sm font-semibold text-gray-900">
            Co-occurrence
          </h2>
        </div>
        <div className="flex gap-1">
          {TYPE_OPTIONS.map((opt) => (
            <button
              key={opt.value}
              onClick={() => setType(opt.value)}
              className={
                type === opt.value
                  ? "badge-blue cursor-pointer"
                  : "badge-gray cursor-pointer hover:bg-gray-200 transition-colors"
              }
            >
              {opt.label}
            </button>
          ))}
        </div>
      </div>

      {loading && (
        <div className="flex items-center justify-center py-12">
          <Spinner />
        </div>
      )}

      {error && (
        <p className="text-sm text-red-500 py-4">{error}</p>
      )}

      {data && data.length === 0 && !loading && (
        <p className="text-sm text-gray-400 py-8 text-center">
          No co-occurrence data available
        </p>
      )}

      {data && data.length > 0 && !loading && (
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b">
                <th className="text-left py-2 pr-4 text-xs font-medium text-gray-500 uppercase tracking-wider">
                  {type === "method-dataset" ? "Method" : "Method"}
                </th>
                <th className="px-4 py-2" />
                <th className="text-left py-2 px-4 text-xs font-medium text-gray-500 uppercase tracking-wider">
                  {type === "method-dataset" ? "Dataset" : "Task"}
                </th>
                <th className="text-right py-2 pl-4 text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Count
                </th>
              </tr>
            </thead>
            <tbody>
              {data.map((row, i) => (
                <tr
                  key={i}
                  className="border-b last:border-0 hover:bg-gray-50 transition-colors"
                >
                  <td className="py-2.5 pr-4">
                    <span className="badge-purple">{row.entity_a}</span>
                  </td>
                  <td className="px-4 py-2.5 text-center">
                    <ArrowRight className="w-3.5 h-3.5 text-gray-300 inline" />
                  </td>
                  <td className="py-2.5 px-4">
                    <span className="badge-green">{row.entity_b}</span>
                  </td>
                  <td className="py-2.5 pl-4 text-right">
                    <span className="text-sm font-medium text-gray-700">
                      {row.co_count}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
