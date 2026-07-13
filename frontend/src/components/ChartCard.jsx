import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Line,
  LineChart,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

// Enough distinct slices for a pie chart before colors repeat; picked to
// read well against the app's light paper background.
const PIE_COLORS = [
  "#336791",
  "#6f9bc2",
  "#b26a00",
  "#1f7a4d",
  "#93588f",
  "#b3261e",
  "#5c6b80",
];

// Renders one `plot_chart` tool result. `spec` is exactly what
// core/charts.py returns: { chart_type, x, y, title, data }.
export default function ChartCard({ spec }) {
  const { chart_type: chartType, x, y, title, data } = spec;

  if (!data || data.length === 0) {
    return <div className="chart-card chart-empty">Chart query returned no rows.</div>;
  }

  return (
    <div className="chart-card">
      {title && <div className="chart-title">{title}</div>}
      <ResponsiveContainer width="100%" height={260}>
        {chartType === "line" ? (
          <LineChart data={data} margin={{ top: 8, right: 16, left: 0, bottom: 0 }}>
            <CartesianGrid stroke="var(--line)" strokeDasharray="3 3" />
            <XAxis dataKey={x} tick={{ fontSize: 12 }} stroke="var(--muted)" />
            <YAxis tick={{ fontSize: 12 }} stroke="var(--muted)" />
            <Tooltip />
            <Line type="monotone" dataKey={y} stroke="var(--accent)" strokeWidth={2} dot={false} />
          </LineChart>
        ) : chartType === "pie" ? (
          <PieChart>
            <Tooltip />
            <Pie data={data} dataKey={y} nameKey={x} outerRadius={95} label>
              {data.map((_, i) => (
                <Cell key={i} fill={PIE_COLORS[i % PIE_COLORS.length]} />
              ))}
            </Pie>
          </PieChart>
        ) : (
          <BarChart data={data} margin={{ top: 8, right: 16, left: 0, bottom: 0 }}>
            <CartesianGrid stroke="var(--line)" strokeDasharray="3 3" />
            <XAxis dataKey={x} tick={{ fontSize: 12 }} stroke="var(--muted)" />
            <YAxis tick={{ fontSize: 12 }} stroke="var(--muted)" />
            <Tooltip />
            <Bar dataKey={y} fill="var(--accent)" radius={[4, 4, 0, 0]} />
          </BarChart>
        )}
      </ResponsiveContainer>
    </div>
  );
}
