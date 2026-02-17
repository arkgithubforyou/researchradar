import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import type { RankedEntity } from "@/lib/types";

interface TopEntitiesChartProps {
  data: RankedEntity[];
  color: string;
}

export default function TopEntitiesChart({ data, color }: TopEntitiesChartProps) {
  // Deduplicate by name, summing counts (in case multiple years returned)
  const byName = new Map<string, number>();
  for (const item of data) {
    byName.set(item.name, (byName.get(item.name) ?? 0) + item.count);
  }

  const chartData = Array.from(byName.entries())
    .sort((a, b) => b[1] - a[1])
    .slice(0, 10)
    .map(([name, count]) => ({
      name: name.length > 22 ? name.slice(0, 20) + "\u2026" : name,
      fullName: name,
      count,
    }));

  if (chartData.length === 0) {
    return (
      <p className="text-sm text-gray-400 py-8 text-center">
        No data available
      </p>
    );
  }

  return (
    <ResponsiveContainer width="100%" height={280}>
      <BarChart
        data={chartData}
        layout="vertical"
        margin={{ top: 4, right: 12, left: 0, bottom: 0 }}
      >
        <CartesianGrid strokeDasharray="3 3" stroke="#f1f3f5" horizontal={false} />
        <XAxis
          type="number"
          tick={{ fontSize: 11, fill: "#868e96" }}
          axisLine={false}
          tickLine={false}
        />
        <YAxis
          type="category"
          dataKey="name"
          width={130}
          tick={{ fontSize: 11, fill: "#495057" }}
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
          formatter={(value: number) => [
            value.toLocaleString(),
            "Papers",
          ]}
        />
        <Bar
          dataKey="count"
          fill={color}
          radius={[0, 4, 4, 0]}
          maxBarSize={20}
          fillOpacity={0.85}
        />
      </BarChart>
    </ResponsiveContainer>
  );
}
