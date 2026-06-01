import { useEffect, useMemo, useState, type MouseEvent } from "react";
import {
  api,
  type DatasetSummary,
  type FeatureAnalyticsDistributionResponse,
  type FeatureAnalyticsObject,
  objectThumbnailUrl,
  type PipelineInfo,
} from "../api/client";

type FilterState = {
  dataset: string;
  session: string;
  labels: string;
  superclass: string;
  validationStatus: string;
  split: string;
  pipeline: string;
  calibration: string;
  dateFrom: string;
  dateTo: string;
};

const DEFAULT_FILTERS: FilterState = {
  dataset: "",
  session: "",
  labels: "",
  superclass: "",
  validationStatus: "",
  split: "",
  pipeline: "",
  calibration: "",
  dateFrom: "",
  dateTo: "",
};

const COLORS = ["#1d4ed8", "#dc2626", "#0f766e", "#9333ea", "#f97316", "#374151"];
const DEFAULT_CHART_WIDTH_PX = 860;

function compactNumber(value: number): string {
  if (!Number.isFinite(value)) return "-";
  const abs = Math.abs(value);
  if (abs >= 100) return value.toFixed(0);
  if (abs >= 10) return value.toFixed(1).replace(/\.0$/, "");
  if (abs >= 1) return value.toFixed(2).replace(/0+$/, "").replace(/\.$/, "");
  return value.toFixed(4).replace(/0+$/, "").replace(/\.$/, "");
}

function adaptiveTickIndexes(edges: number[], binCount: number): number[] {
  if (!edges.length) return [];
  const targetTickCount = Math.max(4, Math.min(10, Math.floor(DEFAULT_CHART_WIDTH_PX / 120)));
  const step = Math.max(1, Math.ceil(binCount / targetTickCount));
  const indexes: number[] = [];
  for (let idx = 0; idx < edges.length; idx += step) indexes.push(idx);
  const lastIdx = edges.length - 1;
  if (!indexes.includes(lastIdx)) indexes.push(lastIdx);
  return indexes;
}

function toQuery(filters: FilterState): Record<string, string> {
  return {
    datasets: filters.dataset,
    sessions: filters.session,
    labels: filters.labels,
    superclass: filters.superclass,
    validation_status: filters.validationStatus,
    split: filters.split,
    pipeline: filters.pipeline,
    calibration: filters.calibration,
    date_from: filters.dateFrom,
    date_to: filters.dateTo,
  };
}

export default function FeatureAnalyticsPage() {
  const [datasets, setDatasets] = useState<DatasetSummary[]>([]);
  const [pipelines, setPipelines] = useState<PipelineInfo[]>([]);
  const [features, setFeatures] = useState<Array<{ feature_key: string; display_name: string; semantic_group: string; unit?: string | null; description?: string | null; source_stage?: string | null }>>([]);
  const [filters, setFilters] = useState<FilterState>(DEFAULT_FILTERS);
  const [featureKey, setFeatureKey] = useState("");
  const [groupBy, setGroupBy] = useState<"superclass" | "label">("superclass");
  const [mode, setMode] = useState<"count" | "density">("count");
  const [binCount, setBinCount] = useState(24);
  const [distribution, setDistribution] = useState<FeatureAnalyticsDistributionResponse | null>(null);
  const [selectedRange, setSelectedRange] = useState<{ min: number; max: number } | null>(null);
  const [showBinEdgeLabels, setShowBinEdgeLabels] = useState(false);
  const [objects, setObjects] = useState<FeatureAnalyticsObject[]>([]);
  const [hoverPreview, setHoverPreview] = useState<{
    x: number;
    y: number;
    src: string;
    item: FeatureAnalyticsObject;
  } | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    void Promise.all([api.datasets(), api.pipelines()])
      .then(([ds, pls]) => {
        setDatasets(ds);
        setPipelines(pls);
      })
      .catch(() => {
        setDatasets([]);
        setPipelines([]);
      });
  }, []);

  useEffect(() => {
    setLoading(true);
    setError(null);
    void api
      .featureAnalyticsFeatures(toQuery(filters))
      .then((res) => {
        setFeatures(res.feature_definitions);
        if (!featureKey && res.feature_definitions[0]?.feature_key) {
          setFeatureKey(res.feature_definitions[0].feature_key);
        }
      })
      .catch((err: unknown) => setError(err instanceof Error ? err.message : "Failed loading feature definitions"))
      .finally(() => setLoading(false));
  }, [filters.dataset, filters.session, filters.validationStatus, filters.pipeline, filters.superclass, filters.labels, filters.split, filters.calibration, filters.dateFrom, filters.dateTo]);

  useEffect(() => {
    if (!featureKey) {
      setDistribution(null);
      return;
    }
    setLoading(true);
    setError(null);
    void api
      .featureAnalyticsDistributions({ ...toQuery(filters), feature_key: featureKey, group_by: groupBy, bins: binCount, mode })
      .then((res) => {
        setDistribution(res);
        setSelectedRange(null);
      })
      .catch((err: unknown) => setError(err instanceof Error ? err.message : "Failed loading distributions"))
      .finally(() => setLoading(false));
  }, [filters, featureKey, groupBy, binCount, mode]);

  useEffect(() => {
    if (!selectedRange || !featureKey) {
      setObjects([]);
      return;
    }
    void api
      .featureAnalyticsObjects({ ...toQuery(filters), feature_key: featureKey, min_value: selectedRange.min, max_value: selectedRange.max, limit: 120 })
      .then((res) => setObjects(res.objects))
      .catch(() => setObjects([]));
  }, [selectedRange, featureKey, filters]);

  const featureMeta = useMemo(() => features.find((item) => item.feature_key === featureKey) ?? null, [features, featureKey]);
  const maxY = useMemo(() => {
    if (!distribution) return 0;
    let max = 0;
    for (const group of distribution.groups) {
      for (const value of group.bins) {
        max = Math.max(max, Number(value) || 0);
      }
    }
    return max;
  }, [distribution]);
  const tickIndexes = useMemo(() => {
    if (!distribution?.range?.edges?.length) return [];
    return adaptiveTickIndexes(distribution.range.edges, binCount);
  }, [distribution, binCount]);

  return (
    <main className="page feature-analytics-page">
      <section className="feature-analytics-layout">
        <aside className="feature-analytics-sidebar">
          <h2>Filters</h2>
          <label>Dataset<select value={filters.dataset} onChange={(event) => setFilters((prev) => ({ ...prev, dataset: event.target.value }))}><option value="">All</option>{datasets.map((item) => <option key={item.id} value={item.id}>{item.name}</option>)}</select></label>
          <label>Session<input value={filters.session} onChange={(event) => setFilters((prev) => ({ ...prev, session: event.target.value }))} placeholder="session_id" /></label>
          <label>Labels<input value={filters.labels} onChange={(event) => setFilters((prev) => ({ ...prev, labels: event.target.value }))} placeholder="comma-separated" /></label>
          <label>Superclass<input value={filters.superclass} onChange={(event) => setFilters((prev) => ({ ...prev, superclass: event.target.value }))} placeholder="BALL_GOOD,..." /></label>
          <label>Validation<input value={filters.validationStatus} onChange={(event) => setFilters((prev) => ({ ...prev, validationStatus: event.target.value }))} placeholder="approved/unreviewed" /></label>
          <label>Split<input value={filters.split} onChange={(event) => setFilters((prev) => ({ ...prev, split: event.target.value }))} placeholder="train/test" /></label>
          <label>Pipeline<select value={filters.pipeline} onChange={(event) => setFilters((prev) => ({ ...prev, pipeline: event.target.value }))}><option value="">All</option>{pipelines.map((item) => <option key={item.id} value={item.id}>{item.display_name}</option>)}</select></label>
          <label>Calibration<input value={filters.calibration} onChange={(event) => setFilters((prev) => ({ ...prev, calibration: event.target.value }))} placeholder="calibration id" /></label>
          <label>Date from<input type="date" value={filters.dateFrom} onChange={(event) => setFilters((prev) => ({ ...prev, dateFrom: event.target.value }))} /></label>
          <label>Date to<input type="date" value={filters.dateTo} onChange={(event) => setFilters((prev) => ({ ...prev, dateTo: event.target.value }))} /></label>
        </aside>

        <section className="feature-analytics-main">
          <header className="feature-analytics-controls">
            <label>Feature<select value={featureKey} onChange={(event) => setFeatureKey(event.target.value)}>{features.map((item) => <option key={item.feature_key} value={item.feature_key}>{item.semantic_group} • {item.display_name}</option>)}</select></label>
            <label>Grouping<select value={groupBy} onChange={(event) => setGroupBy(event.target.value as "superclass" | "label")}><option value="superclass">Superclass</option><option value="label">Label</option></select></label>
            <label>Mode<select value={mode} onChange={(event) => setMode(event.target.value as "count" | "density")}><option value="count">Count</option><option value="density">Normalized density</option></select></label>
            <label>Bins<input type="range" min={8} max={64} value={binCount} onChange={(event) => setBinCount(Number(event.target.value))} /><span>{binCount}</span></label>
            <label className="feature-inline-toggle"><input type="checkbox" checked={showBinEdgeLabels} onChange={(event) => setShowBinEdgeLabels(event.target.checked)} />Show bin edge labels</label>
          </header>

          <section className="feature-analytics-plot">
            {loading ? <div className="empty-state">Loading feature analytics...</div> : null}
            {!loading && error ? <div className="empty-state"><strong>{error}</strong></div> : null}
            {!loading && !error && distribution?.groups.length ? (
              <div>
                <div className="feature-legend">{distribution.groups.map((group, idx) => <span key={group.group}><i style={{ backgroundColor: COLORS[idx % COLORS.length] }} />{group.group}</span>)}</div>
                <div className="feature-histogram-grid">
                  {distribution.groups.map((group, groupIdx) => (
                    <div key={group.group} className="feature-histogram-row">
                      {group.bins.map((value, idx) => {
                        const edgeStart = distribution.range?.edges[idx] ?? 0;
                        const edgeEnd = distribution.range?.edges[idx + 1] ?? edgeStart;
                        const density = group.count > 0 ? Number(value) / group.count : 0;
                        const pct = maxY ? (Number(value) / maxY) * 100 : 0;
                        return (
                          <button
                            type="button"
                            key={`${group.group}_${idx}`}
                            className="feature-bin"
                            title={`${group.group} | ${compactNumber(edgeStart)} - ${compactNumber(edgeEnd)} | count ${compactNumber(Number(value))}${mode === "density" ? ` | density ${compactNumber(Number(value))}` : ` | density ${compactNumber(density)}`}`}
                            onClick={() => setSelectedRange({ min: edgeStart, max: edgeEnd })}
                          >
                            <span style={{ height: `${Math.max(2, pct)}%`, backgroundColor: COLORS[groupIdx % COLORS.length] }} />
                          </button>
                        );
                      })}
                    </div>
                  ))}
                </div>
                {distribution.range?.edges?.length ? (
                  <div className="feature-axis" aria-label="Feature value axis">
                    {tickIndexes.map((tickIndex) => {
                      const leftPct = distribution.range?.edges?.length && distribution.range.edges.length > 1
                        ? (tickIndex / (distribution.range.edges.length - 1)) * 100
                        : 0;
                      const tickValue = distribution.range?.edges?.[tickIndex] ?? 0;
                      return (
                        <span key={`tick_${tickIndex}`} className="feature-axis-tick" style={{ left: `${leftPct}%` }}>
                          {compactNumber(tickValue)}
                        </span>
                      );
                    })}
                    {showBinEdgeLabels ? (
                      <div className="feature-axis-edge-grid">
                        {distribution.range.edges.map((edgeValue, edgeIdx) => (
                          <span key={`edge_${edgeIdx}`} className="feature-axis-edge-label">{compactNumber(edgeValue)}</span>
                        ))}
                      </div>
                    ) : null}
                  </div>
                ) : null}
              </div>
            ) : null}
          </section>

          <section className="feature-linked-objects">
            <h3>Linked samples {selectedRange ? `[${selectedRange.min.toFixed(4)}, ${selectedRange.max.toFixed(4)}]` : ""}</h3>
            {!objects.length ? <div className="empty-state">Click a histogram bin to inspect matching objects.</div> : null}
            {objects.length ? (
              <div className="feature-linked-table-wrap">
                <table>
                  <thead><tr><th>Preview</th><th>Take</th><th>Object</th><th>Superclass</th><th>Feature</th><th>Open</th></tr></thead>
                  <tbody>
                    {objects.map((item) => (
                      <tr key={`${item.take_id}_${item.object_id}`}>
                        <td>
                          <ObjectThumbCell
                            item={item}
                            onHover={(payload) => setHoverPreview(payload)}
                            onLeave={() => setHoverPreview(null)}
                          />
                        </td>
                        <td>{item.take_id}</td>
                        <td>{item.object_id}</td>
                        <td>{item.superclass ?? "-"}</td>
                        <td>{compactNumber(item.feature_value)}</td>
                        <td><a href={`/takes/${encodeURIComponent(item.take_id)}`}>Open take</a></td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : null}
            {hoverPreview ? (
              <div
                className="feature-thumb-hover-card"
                style={{ left: `${hoverPreview.x + 14}px`, top: `${hoverPreview.y + 14}px` }}
              >
                <img alt={`Object ${hoverPreview.item.object_id} preview`} src={hoverPreview.src} loading="lazy" />
                <small>{hoverPreview.item.take_id} • object {hoverPreview.item.object_id}</small>
                <small>{hoverPreview.item.superclass ?? "UNKNOWN"} • {compactNumber(hoverPreview.item.feature_value)}</small>
              </div>
            ) : null}
          </section>
        </section>

        <aside className="feature-analytics-inspector">
          <h2>Feature inspector</h2>
          {!featureMeta ? <div className="empty-state">Select a feature.</div> : (
            <div className="inspector-list">
              <p><strong>{featureMeta.display_name}</strong></p>
              <p>Key: <code>{featureMeta.feature_key}</code></p>
              <p>Group: {featureMeta.semantic_group}</p>
              <p>Unit: {featureMeta.unit ?? "-"}</p>
              <p>Source stage: {featureMeta.source_stage ?? "-"}</p>
              <p>{featureMeta.description ?? "No description."}</p>
              <hr />
              <p>Count: {distribution?.stats.count ?? 0}</p>
              <p>Missing %: {(distribution?.stats.missing_pct ?? 0).toFixed(2)}%</p>
              <p>Min / Max: {distribution?.stats.min ?? "-"} / {distribution?.stats.max ?? "-"}</p>
              <p>Mean / Std: {distribution?.stats.mean ?? "-"} / {distribution?.stats.std ?? "-"}</p>
            </div>
          )}
        </aside>
      </section>
    </main>
  );
}

function ObjectThumbCell({
  item,
  onHover,
  onLeave,
}: {
  item: FeatureAnalyticsObject;
  onHover: (payload: { x: number; y: number; src: string; item: FeatureAnalyticsObject }) => void;
  onLeave: () => void;
}) {
  const [failed, setFailed] = useState(false);
  const src = objectThumbnailUrl(item.take_id, item.object_id, 56);

  function handleMove(event: MouseEvent<HTMLDivElement>) {
    onHover({ x: event.clientX, y: event.clientY, src, item });
  }

  return (
    <div
      className="feature-object-thumb"
      onMouseEnter={handleMove}
      onMouseMove={handleMove}
      onMouseLeave={onLeave}
    >
      {!failed ? (
        <img
          alt={`Object ${item.object_id} thumbnail`}
          src={src}
          loading="lazy"
          onError={() => setFailed(true)}
        />
      ) : (
        <div className="feature-object-thumb-placeholder" />
      )}
    </div>
  );
}
