import type { ProcessingResult } from "../api/client";
import { objectDisplayId } from "./objectSelectionModel";

type Props = {
  objects: ProcessingResult["objects"];
  selectedObjectId?: number | null;
  hoveredObjectId?: number | null;
  onSelectObject?: (objectId: number) => void;
  onHoverObject?: (objectId: number | null) => void;
};

function fmt(value: number | null | undefined, digits = 2) {
  return typeof value === "number" ? value.toFixed(digits) : "-";
}

function vector(value: [number, number, number] | null | undefined) {
  return value ? value.map((item) => item.toFixed(1)).join(", ") : "-";
}

function fmtUnknown(value: unknown, digits = 1) {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric.toFixed(digits) : "-";
}

function heightRange(value: Record<string, unknown> | null | undefined) {
  if (!value) return "-";
  const min = value.min ?? value.min_height_mm;
  const mean = value.mean ?? value.mean_height_mm;
  const max = value.max ?? value.max_height_mm;
  const p95 = value.p95 ?? value.p95_height_mm;
  const median = value.median ?? value.median_height_mm;
  const compact = [min, mean, max].some((item) => Number.isFinite(Number(item)))
    ? `${fmtUnknown(min)} / ${fmtUnknown(mean)} / ${fmtUnknown(max)}`
    : "-";
  return p95 == null && median == null ? compact : `${compact} (median ${fmtUnknown(median)}, p95 ${fmtUnknown(p95)})`;
}

export default function ObjectTable({ objects, selectedObjectId = null, hoveredObjectId = null, onSelectObject, onHoverObject }: Props) {
  if (objects.length === 0) {
    return <div className="empty-state">No objects reported.</div>;
  }
  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            <th>ID</th>
            <th>Class</th>
            <th>Subclass</th>
            <th>Points</th>
            <th>Confidence</th>
            <th>Center (mm)</th>
            <th>Dimensions (mm)</th>
            <th>Diameter estimate (mm)</th>
            <th>Diameter (mm)</th>
            <th>Height above belt (mm)</th>
            <th>Fraction inside belt</th>
            <th>Filter status</th>
            <th>Filter reason</th>
            <th>Sphericity</th>
            <th>Fit RMSE (mm)</th>
          </tr>
        </thead>
        <tbody>
          {objects.map((object) => (
            <tr
              className={object.object_id === selectedObjectId ? "selected-object-row" : object.object_id === hoveredObjectId ? "hovered-object-row" : ""}
              key={object.object_id}
              onClick={() => onSelectObject?.(object.object_id)}
              onMouseEnter={() => onHoverObject?.(object.object_id)}
              onMouseLeave={() => onHoverObject?.(null)}
            >
              <td>{objectDisplayId(object.object_id)}</td>
              <td>{object.class_name}</td>
              <td>{object.subclass_label ?? "-"}</td>
              <td>{object.point_count?.toLocaleString() ?? "-"}</td>
              <td>{object.confidence == null ? "-" : `${Math.round(object.confidence * 100)}%`}</td>
              <td>{vector(object.center_mm)}</td>
              <td>{vector(object.dimensions_mm)}</td>
              <td>{fmt(object.diameter_estimate_mm)}</td>
              <td>{fmt(object.diameter_mm)}</td>
              <td>{heightRange(object.height_above_belt_mm)}</td>
              <td>{fmt(object.fraction_points_inside_belt, 3)}</td>
              <td>{object.filter_status ?? "-"}</td>
              <td>{object.filter_reason ?? "-"}</td>
              <td>{fmt(object.sphericity_score, 3)}</td>
              <td>{fmt(object.fit_rmse_mm, 3)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
