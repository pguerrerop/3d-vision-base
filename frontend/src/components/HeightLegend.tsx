import { cssGradientForColorMap, type NamedColorMap } from "./colormap";
import type { HeightColorMapping } from "./heightSemantics";

type HeightLegendProps = {
  // Canonical input. When provided, the legend treats this as the single source
  // of truth: the gradient is rendered from the actual LUT, min/max labels show
  // the color-scale bounds, and `direction` ONLY affects the tick labels (the
  // colormap itself is always data-native: valueMin -> cool, valueMax -> hot).
  colorMapping?: HeightColorMapping | null;
  // Legacy props (still supported for callers not yet migrated).
  semanticField?: string;
  semanticLabel?: string;
  units?: string;
  minValue?: number | null;
  maxValue?: number | null;
  colorMap?: string;
  positiveDirection?: string;
  authoritative?: boolean;
  compact?: boolean;
  // When true, renders the "Higher value / Lower value"-style direction labels
  // under the gradient.
  showTicks?: boolean;
  // When > 0, renders that many numeric tick marks alongside the gradient,
  // evenly spaced between colorScaleMin and colorScaleMax. Defaults to 5.
  tickCount?: number;
  percentileLabels?: { p2?: number | null; p98?: number | null };
};

export default function HeightLegend({
  colorMapping,
  semanticField,
  semanticLabel,
  units,
  minValue,
  maxValue,
  colorMap,
  positiveDirection,
  authoritative,
  compact = false,
  showTicks = false,
  tickCount = 5,
  percentileLabels,
}: HeightLegendProps) {
  const effectiveSemanticField = colorMapping?.semanticField ?? semanticField ?? "unknown";
  const effectiveUnits = colorMapping?.units ?? units ?? "mm";
  const effectiveMin = colorMapping?.colorScaleMin ?? (Number.isFinite(minValue ?? NaN) ? Number(minValue) : null);
  const effectiveMax = colorMapping?.colorScaleMax ?? (Number.isFinite(maxValue ?? NaN) ? Number(maxValue) : null);
  const effectiveColorMap: NamedColorMap = (colorMapping?.colorMap ?? colorMap ?? "turbo") as NamedColorMap;
  const effectiveDirection: "higher_is_hotter" | "lower_is_hotter" = colorMapping?.direction ?? "higher_is_hotter";
  const effectiveSource = colorMapping?.source ?? null;
  const title = semanticLabel?.trim() || effectiveSemanticField;
  const directionHigh = positiveDirectionLabel(positiveDirection ?? effectiveSemanticField, true, effectiveDirection);
  const directionLow = positiveDirectionLabel(positiveDirection ?? effectiveSemanticField, false, effectiveDirection);
  const modeClass = compact ? "height-legend compact" : "height-legend full";
  // CSS gradient is generated from the SAME named colormap the renderer uses,
  // so the legend can never visually disagree with the rendered preview.
  const gradient = cssGradientForColorMap(effectiveColorMap, 16, effectiveDirection);
  const ticks = buildNumericTicks(effectiveMin, effectiveMax, tickCount, effectiveDirection, effectiveUnits);
  return (
    <section className={modeClass} aria-label="Height legend">
      <header className="height-legend-header">
        <strong>{title}</strong>
        <small>{authoritative ? "Authoritative" : "Debug/preview"}</small>
      </header>
      <div className="height-legend-body">
        <span className="height-legend-limit">{effectiveMax == null ? "-" : `${Number(effectiveMax).toFixed(2)} ${effectiveUnits}`}</span>
        <div className="height-legend-bar">
          <div className="height-legend-gradient" style={{ background: gradient }} aria-hidden="true" />
          {ticks.length > 0 && (
            <ul className="height-legend-tick-list" aria-label="Color scale tick values">
              {ticks.map((tick) => (
                <li
                  key={`${tick.position}-${tick.label}`}
                  className="height-legend-tick"
                  style={{ top: `${tick.position}%` }}
                >
                  <span className="height-legend-tick-mark" aria-hidden="true" />
                  <span className="height-legend-tick-label">{tick.label}</span>
                </li>
              ))}
            </ul>
          )}
        </div>
        <span className="height-legend-limit">{effectiveMin == null ? "-" : `${Number(effectiveMin).toFixed(2)} ${effectiveUnits}`}</span>
      </div>
      {showTicks && (
        <div className="height-legend-ticks">
          <small>{directionHigh}</small>
          <small>{directionLow}</small>
        </div>
      )}
      {!!percentileLabels && (
        <div className="height-legend-percentiles">
          <small>p2: {fmtOptional(percentileLabels.p2)}</small>
          <small>p98: {fmtOptional(percentileLabels.p98)}</small>
        </div>
      )}
      {effectiveSource && (
        <div className="height-legend-source">
          <small>colormap: {effectiveColorMap}</small>
          <small>source: {effectiveSource}</small>
        </div>
      )}
    </section>
  );
}

type NumericTick = { position: number; label: string };

function buildNumericTicks(
  min: number | null,
  max: number | null,
  count: number,
  direction: "higher_is_hotter" | "lower_is_hotter",
  units: string,
): NumericTick[] {
  if (count <= 0) return [];
  if (min == null || max == null || !Number.isFinite(min) || !Number.isFinite(max) || max <= min) return [];
  const steps = Math.max(2, Math.floor(count));
  const span = max - min;
  const precision = pickPrecision(span);
  const out: NumericTick[] = [];
  for (let i = 0; i < steps; i += 1) {
    const tFromBottom = i / (steps - 1);
    const value = min + tFromBottom * span;
    // Gradient is rendered top->bottom (180deg). For higher_is_hotter, top = max
    // and bottom = min, so the largest value sits at 0% and the smallest at 100%.
    const position = direction === "higher_is_hotter"
      ? (1 - tFromBottom) * 100
      : tFromBottom * 100;
    out.push({ position, label: `${value.toFixed(precision)} ${units}` });
  }
  return out;
}

function pickPrecision(span: number): number {
  if (span >= 100) return 0;
  if (span >= 10) return 1;
  if (span >= 1) return 2;
  return 3;
}

function positiveDirectionLabel(direction: string | undefined, high: boolean, mappingDirection: "higher_is_hotter" | "lower_is_hotter"): string {
  const key = String(direction ?? "").toLowerCase();
  const hotter = mappingDirection === "higher_is_hotter" ? high : !high;
  if (key.includes("belt") || key.includes("height_above_belt")) {
    return hotter ? "Higher above belt" : "Belt/lower";
  }
  if (key.includes("signed_distance") || key.includes("plane_signed_distance")) {
    return hotter ? "Positive distance" : "Negative distance";
  }
  if (key.includes("raw_sensor_z")) {
    return hotter ? "Higher sensor Z" : "Lower sensor Z";
  }
  return hotter ? "Higher value" : "Lower value";
}

function fmtOptional(value: number | null | undefined): string {
  return Number.isFinite(value ?? NaN) ? `${Number(value).toFixed(2)}` : "-";
}
