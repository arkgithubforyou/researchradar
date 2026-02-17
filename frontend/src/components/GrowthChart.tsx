import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import type { GrowthPoint } from "@/lib/types";

interface GrowthChartProps {
  data: GrowthPoint[];
}

export default function GrowthChart({ data }: GrowthChartProps) {
  return (
    <ResponsiveContainer width="100%" height={260}>
      <AreaChart data={data} margin={{ top: 4, right: 4, left: -20, bottom: 0 }}>
        <defs>
          <linearGradient id="growthGradient" x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%" stopColor="#4263eb" stopOpacity={0.15} />
            <stop offset="95%" stopColor="#4263eb" stopOpacity={0.01} />
          </linearGradient>
        </defs>
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
          formatter={(value: number) => [value.toLocaleString(), "Papers"]}
          labelFormatter={(label: number) => `Year ${label}`}
        />
        <Area
          type="monotone"
          dataKey="paper_count"
          stroke="#4263eb"
          strokeWidth={2}
          fill="url(#growthGradient)"
          dot={{ r: 3, fill: "#4263eb", strokeWidth: 0 }}
          activeDot={{ r: 5, strokeWidth: 2, stroke: "#fff" }}
        />
      </AreaChart>
    </ResponsiveContainer>
  );
}
