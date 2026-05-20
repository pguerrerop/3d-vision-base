import type { ProcessingResult } from "../api/client";

type Stats = NonNullable<ProcessingResult["input_stats"]>;

type Props = {
  stats?: Stats | null;
};

function vector(value?: [number, number, number] | null) {
  if (!value) {
    return "-";
  }
  return value.map((item) => item.toFixed(2)).join(", ");
}

function bytes(value?: number | null) {
  if (typeof value !== "number") {
    return "-";
  }
  if (value < 1024) {
    return `${value} B`;
  }
  if (value < 1024 * 1024) {
    return `${(value / 1024).toFixed(1)} KB`;
  }
  return `${(value / (1024 * 1024)).toFixed(2)} MB`;
}

export default function PointCloudStatsCard({ stats }: Props) {
  if (!stats) {
    return <div className="stats-card empty-stats">Point cloud stats are not available.</div>;
  }

  return (
    <section className="stats-card">
      <div>
        <span>Points</span>
        <strong>{stats.point_count.toLocaleString()}</strong>
      </div>
      <div>
        <span>File size</span>
        <strong>{bytes(stats.file_size_bytes)}</strong>
      </div>
      <div>
        <span>Has colors</span>
        <strong>{stats.has_colors ? "Yes" : "No"}</strong>
      </div>
      <div>
        <span>Has normals</span>
        <strong>{stats.has_normals ? "Yes" : "No"}</strong>
      </div>
      <div>
        <span>Min bound (mm)</span>
        <strong>{vector(stats.min_bound)}</strong>
      </div>
      <div>
        <span>Max bound (mm)</span>
        <strong>{vector(stats.max_bound)}</strong>
      </div>
      <div className="wide">
        <span>Extent (mm)</span>
        <strong>{vector(stats.extent)}</strong>
      </div>
    </section>
  );
}
