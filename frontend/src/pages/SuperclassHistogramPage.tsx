import { useEffect, useMemo, useState } from "react";
import { api, type DatasetSummary, type PipelineInfo } from "../api/client";

type SuperclassFeatureKey = "max_h" | "ecc" | "sph3d" | "flat" | "rough" | "footprint_roundness";
type SuperclassKey = "BALL_GOOD" | "BALL_SCRAP" | "SCRAP" | "UNKNOWN";
type FeatureHistogramStats = {
  count: number;
  min: number;
  max: number;
  bins: number[];
};
type SuperclassFeatureState = Record<SuperclassKey, Partial<Record<SuperclassFeatureKey, FeatureHistogramStats>>>;
type SuperclassCountState = Record<SuperclassKey, number>;
type DatasetFeatureHistogramState = {
  loading: boolean;
  error: string | null;
  takeCount: number;
  objectCount: number;
  superclassCounts: SuperclassCountState;
  statsBySuperclass: SuperclassFeatureState;
};

const SUPERCLASS_FEATURES: Array<{ key: SuperclassFeatureKey; label: string }> = [
  { key: "max_h", label: "max_h" },
  { key: "ecc", label: "ecc" },
  { key: "sph3d", label: "sph3d" },
  { key: "flat", label: "flat" },
  { key: "rough", label: "rough" },
  { key: "footprint_roundness", label: "footprint_roundness" },
];

const SUPERCLASS_ORDER: SuperclassKey[] = ["BALL_GOOD", "BALL_SCRAP", "SCRAP", "UNKNOWN"];
const SUPERCLASS_LABELS: Record<SuperclassKey, string> = {
  BALL_GOOD: "BALL_GOOD",
  BALL_SCRAP: "BALL_SCRAP",
  SCRAP: "SCRAP",
  UNKNOWN: "UNKNOWN",
};

function emptySuperclassFeatureState(): SuperclassFeatureState {
  return {
    BALL_GOOD: {},
    BALL_SCRAP: {},
    SCRAP: {},
    UNKNOWN: {},
  };
}

function emptySuperclassCountState(): SuperclassCountState {
  return {
    BALL_GOOD: 0,
    BALL_SCRAP: 0,
    SCRAP: 0,
    UNKNOWN: 0,
  };
}

function toFinite(value: unknown): number | null {
  const n = Number(value);
  return Number.isFinite(n) ? n : null;
}

function histogram(values: number[], binCount = 24): FeatureHistogramStats | null {
  if (!values.length) return null;
  const min = Math.min(...values);
  const max = Math.max(...values);
  const bins = Array.from({ length: binCount }, () => 0);
  if (Math.abs(max - min) < 1e-12) {
    bins[0] = values.length;
    return { count: values.length, min, max, bins };
  }
  const width = (max - min) / binCount;
  for (const value of values) {
    const clamped = Math.max(min, Math.min(max, value));
    const idx = Math.min(binCount - 1, Math.floor((clamped - min) / width));
    bins[idx] += 1;
  }
  return { count: values.length, min, max, bins };
}

function classifySuperclass(item: Record<string, unknown>): SuperclassKey {
  const superclass = String(item.superclass ?? "").trim().toUpperCase();
  if (superclass === "BALL_GOOD") return "BALL_GOOD";
  if (superclass === "BALL_SCRAP") return "BALL_SCRAP";
  if (superclass === "SCRAP" || superclass === "SCRAP_METAL") return "SCRAP";
  if (superclass === "UNKNOWN") return "UNKNOWN";
  const cg = String(item.class_group ?? "").trim().toLowerCase();
  if (cg.includes("bola buena")) return "BALL_GOOD";
  if (cg.includes("scrap de bola")) return "BALL_SCRAP";
  if (cg.includes("chatarra")) return "SCRAP";
  const label = String(item.label ?? "").trim().toLowerCase();
  if (label.startsWith("bola buena")) return "BALL_GOOD";
  if (label.startsWith("scrap de bola")) return "BALL_SCRAP";
  if (label.startsWith("chatarra")) return "SCRAP";
  return "UNKNOWN";
}

function buildFeatureValuesBySuperclass() {
  const state: Record<SuperclassKey, Record<SuperclassFeatureKey, number[]>> = {
    BALL_GOOD: { max_h: [], ecc: [], sph3d: [], flat: [], rough: [], footprint_roundness: [] },
    BALL_SCRAP: { max_h: [], ecc: [], sph3d: [], flat: [], rough: [], footprint_roundness: [] },
    SCRAP: { max_h: [], ecc: [], sph3d: [], flat: [], rough: [], footprint_roundness: [] },
    UNKNOWN: { max_h: [], ecc: [], sph3d: [], flat: [], rough: [], footprint_roundness: [] },
  };
  return state;
}

export default function SuperclassHistogramPage() {
  const [datasets, setDatasets] = useState<DatasetSummary[]>([]);
  const [pipelines, setPipelines] = useState<PipelineInfo[]>([]);
  const [selectedDataset, setSelectedDataset] = useState("");
  const [selectedPipelineId, setSelectedPipelineId] = useState("");
  const [state, setState] = useState<DatasetFeatureHistogramState>({
    loading: false,
    error: null,
    takeCount: 0,
    objectCount: 0,
    superclassCounts: emptySuperclassCountState(),
    statsBySuperclass: emptySuperclassFeatureState(),
  });
  const [rerunBusy, setRerunBusy] = useState(false);
  const [rerunStatus, setRerunStatus] = useState<string | null>(null);
  const [rerunDone, setRerunDone] = useState<{ ok: number; failed: number; total: number } | null>(null);

  useEffect(() => {
    void Promise.all([api.datasets(), api.pipelines()]).then(([datasetItems, pipelineItems]) => {
      setDatasets(datasetItems);
      setPipelines(pipelineItems);
      if (!selectedDataset && datasetItems[0]?.id) setSelectedDataset(datasetItems[0].id);
      if (!selectedPipelineId) {
        const default25d = pipelineItems.find((item) => item.id === "mining_steel_ball_classification_25d")
          ?? pipelineItems.find((item) => String(item.pipeline_family ?? "").toLowerCase() === "25d")
          ?? pipelineItems[0];
        if (default25d?.id) setSelectedPipelineId(default25d.id);
      }
    }).catch(() => {
      setDatasets([]);
      setPipelines([]);
    });
  }, []);

  async function loadHistograms() {
    if (!selectedDataset) {
      setState({
        loading: false,
        error: null,
        takeCount: 0,
        objectCount: 0,
        superclassCounts: emptySuperclassCountState(),
        statsBySuperclass: emptySuperclassFeatureState(),
      });
      return;
    }
    setState((current) => ({ ...current, loading: true, error: null }));
    try {
      const takes = await api.filteredTakes({ dataset_id: selectedDataset, show_archived: false });
      const details = await Promise.all(takes.map((item) => api.take(item.take_id).catch(() => null)));
      const bySuperclass = buildFeatureValuesBySuperclass();
      const superclassCounts = emptySuperclassCountState();
      let objectCount = 0;
      for (const detail of details) {
        const objects = ((detail?.result?.objects ?? []) as Array<Record<string, unknown>>)
          .concat((detail?.result?.rejected_objects ?? []) as Array<Record<string, unknown>>);
        objectCount += objects.length;
        for (const item of objects) {
          const superclass = classifySuperclass(item);
          superclassCounts[superclass] += 1;
          const maxH = toFinite((item.height_above_belt_mm as Record<string, unknown> | undefined)?.max_height_mm);
          if (maxH != null) bySuperclass[superclass].max_h.push(maxH);
          const ecc = toFinite(item.feature_eccentricity);
          if (ecc != null) bySuperclass[superclass].ecc.push(ecc);
          const sph3d = toFinite(item.feature_sphericity_3d);
          if (sph3d != null) bySuperclass[superclass].sph3d.push(sph3d);
          const flat = toFinite(item.feature_flatness);
          if (flat != null) bySuperclass[superclass].flat.push(flat);
          const rough = toFinite(item.feature_edge_roughness);
          if (rough != null) bySuperclass[superclass].rough.push(rough);
          const footprintRoundness = toFinite(
            item.footprint_roundness
            ?? item.feature_footprint_roundness
            ?? item.sphere_fit
            ?? item.feature_sphere_fit
            ?? item.sphericity_score
          );
          if (footprintRoundness != null) {
            bySuperclass[superclass].footprint_roundness.push(footprintRoundness);
          } else {
            const rmse = toFinite(item.fit_rmse_mm);
            if (rmse != null && rmse >= 0) bySuperclass[superclass].footprint_roundness.push(1 / (1 + rmse));
          }
        }
      }
      const statsBySuperclass = emptySuperclassFeatureState();
      for (const superclass of SUPERCLASS_ORDER) {
        for (const feature of SUPERCLASS_FEATURES) {
          const parsed = histogram(bySuperclass[superclass][feature.key], 24);
          if (parsed) statsBySuperclass[superclass][feature.key] = parsed;
        }
      }
      setState({
        loading: false,
        error: null,
        takeCount: takes.length,
        objectCount,
        superclassCounts,
        statsBySuperclass,
      });
    } catch (err) {
      setState({
        loading: false,
        error: err instanceof Error ? err.message : "Failed to compute dataset feature histograms",
        takeCount: 0,
        objectCount: 0,
        superclassCounts: emptySuperclassCountState(),
        statsBySuperclass: emptySuperclassFeatureState(),
      });
    }
  }

  useEffect(() => {
    void loadHistograms();
  }, [selectedDataset]);

  async function rerunDatasetPipeline() {
    if (!selectedDataset || !selectedPipelineId || rerunBusy) return;
    const ok = window.confirm("Re-run selected pipeline for all takes in this dataset? This may take a while.");
    if (!ok) return;
    setRerunBusy(true);
    setRerunDone(null);
    setRerunStatus("Loading takes...");
    try {
      const takes = await api.filteredTakes({ dataset_id: selectedDataset, show_archived: false });
      let doneOk = 0;
      let doneFailed = 0;
      for (let i = 0; i < takes.length; i += 1) {
        const take = takes[i];
        setRerunStatus(`Re-running ${i + 1}/${takes.length}: ${take.take_id}`);
        try {
          await api.processTake(take.take_id, {
            pipeline_id: selectedPipelineId,
            reprocess: true,
            run_until_stage: null,
            stage_params: null,
          });
          doneOk += 1;
        } catch {
          doneFailed += 1;
        }
      }
      setRerunDone({ ok: doneOk, failed: doneFailed, total: takes.length });
      setRerunStatus(`Completed: ${doneOk}/${takes.length} succeeded, ${doneFailed} failed.`);
      await loadHistograms();
    } finally {
      setRerunBusy(false);
    }
  }

  const selectedDatasetName = useMemo(
    () => datasets.find((item) => item.id === selectedDataset)?.name ?? selectedDataset,
    [datasets, selectedDataset]
  );

  const selectedPipelineName = useMemo(
    () => pipelines.find((item) => item.id === selectedPipelineId)?.display_name ?? selectedPipelineId,
    [pipelines, selectedPipelineId]
  );

  return (
    <main className="operator-page">
      <section className="page-title-block">
        <div className="eyebrow">Analytics</div>
        <h1>Superclass Feature Histograms</h1>
        <p>Per-dataset distributions for classification variables, split by superclass.</p>
      </section>
      <section className="operator-toolbar">
        <label className="field-label">
          Dataset
          <select value={selectedDataset} onChange={(event) => setSelectedDataset(event.target.value)}>
            <option value="">Select dataset</option>
            {datasets.map((dataset) => (
              <option key={dataset.id} value={dataset.id}>{dataset.name}</option>
            ))}
          </select>
        </label>
        <label className="field-label">
          Pipeline
          <select value={selectedPipelineId} onChange={(event) => setSelectedPipelineId(event.target.value)}>
            {pipelines.map((pipeline) => (
              <option key={pipeline.id} value={pipeline.id}>{pipeline.display_name}</option>
            ))}
          </select>
        </label>
        <button type="button" className="primary-button" onClick={() => void rerunDatasetPipeline()} disabled={!selectedDataset || !selectedPipelineId || rerunBusy}>
          {rerunBusy ? "Running..." : "Re-run pipeline for whole dataset"}
        </button>
        <button type="button" onClick={() => void loadHistograms()} disabled={state.loading || rerunBusy}>Refresh histograms</button>
      </section>
      <section className="concept-panel compact studio-run-summary">
        <small>Selected dataset: {selectedDatasetName || "-"}</small>
        <small>Selected pipeline: {selectedPipelineName || "-"}</small>
        {rerunStatus && <small>{rerunStatus}</small>}
        {rerunDone && <small>Last run summary: {rerunDone.ok} succeeded, {rerunDone.failed} failed, total {rerunDone.total}.</small>}
      </section>
      <section className="concept-panel compact studio-run-summary">
        {state.loading && <small>Computing histograms for selected dataset...</small>}
        {state.error && <small>{state.error}</small>}
        {!state.loading && !state.error && (
          <>
            <small>Dataset takes: {state.takeCount} | Objects sampled: {state.objectCount}</small>
            <div className="pipeline-status-grid">
              {SUPERCLASS_ORDER.map((key) => (
                <div key={key}>
                  <span>{SUPERCLASS_LABELS[key]}</span>
                  <strong>{state.superclassCounts[key]}</strong>
                </div>
              ))}
            </div>
            {SUPERCLASS_ORDER.map((superclass) => (
              <section className="dataset-superclass-section" key={superclass}>
                <div className="concept-heading">
                  <span className="eyebrow">Superclass</span>
                  <strong>{SUPERCLASS_LABELS[superclass]}</strong>
                </div>
                <div className="dataset-feature-hist-grid">
                  {SUPERCLASS_FEATURES.map((feature) => {
                    const stats = state.statsBySuperclass[superclass][feature.key];
                    if (!stats) {
                      return (
                        <div className="dataset-feature-hist-card" key={`${superclass}_${feature.key}`}>
                          <span>{feature.label}</span>
                          <strong>No data</strong>
                        </div>
                      );
                    }
                    const maxBin = Math.max(...stats.bins, 1);
                    return (
                      <div className="dataset-feature-hist-card" key={`${superclass}_${feature.key}`}>
                        <span>{feature.label}</span>
                        <strong>{stats.count} samples</strong>
                        <small>{stats.min.toFixed(3)} to {stats.max.toFixed(3)}</small>
                        <div className="dataset-feature-hist-bars">
                          {stats.bins.map((value, idx) => (
                            <div key={`${superclass}_${feature.key}_${idx}`} className="dataset-feature-hist-bar-wrap">
                              <div className="dataset-feature-hist-bar" style={{ height: `${Math.max(4, (value / maxBin) * 82)}px` }} />
                            </div>
                          ))}
                        </div>
                      </div>
                    );
                  })}
                </div>
              </section>
            ))}
          </>
        )}
      </section>
    </main>
  );
}
