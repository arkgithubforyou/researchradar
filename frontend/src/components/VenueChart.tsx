import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from "recharts";
import { CHART_COLORS, venueLabel } from "@/lib/utils";

interface VenueChartProps {
  data: Array<{ venue: string; paper_count: number }>;
}

export default function VenueChart({ data }: VenueChartProps) {
  const sorted = [...data].sort((a, b) => b.paper_count - a.paper_count).slice(0, 8);
  const chartData = sorted.map((d) => ({
    ...d,
    label: venueLabel(d.venue),
  }));

  return (
    <ResponsiveContainer width="100%" height={260}>
      <BarChart
        data={chartData}
        layout="vertical"
        margin={{ top: 4, right: 4, left: 0, bottom: 0 }}
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
          dataKey="label"
          width={80}
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
          formatter={(value: number) => [value.toLocaleString(), "Papers"]}
        />
        <Bar dataKey="paper_count" radius={[0, 4, 4, 0]} maxBarSize={24}>
          {chartData.map((_, i) => (
            <Cell key={i} fill={CHART_COLORS[i % CHART_COLORS.length]} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}
