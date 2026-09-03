import { useEffect, useMemo, useState } from "react";
import {
  api,
  type MLBackendInfo,
  type MLDatasetInsights,
  type MLDatasetSummary,
  type MLDeploymentSummary,
  type MLExperimentSummary,
  type MLSmokeSummary,
} from "../api/client";

type TabKey =
  | "overview"
  | "datasets"
  | "labels"
  | "features"
  | "experiments"
  | "evaluation"
  | "registry"
  | "deployment"
  | "runtime_health";

type BadgeTone =
  | "healthy"
  | "warning"
  | "failed"
  | "active"
  | "fallback"
  | "heuristic"
  | "incompatible"
  | "invalid"
  | "superseded"
  | "neutral";

const TABS: Array<{ key: TabKey; label: string }> = [
  { key: "overview", label: "Overview" },
  { key: "datasets", label: "Datasets" },
  { key: "labels", label: "Labels" },
  { key: "features", label: "Features" },
  { key: "experiments", label: "Experiments" },
  { key: "evaluation", label: "Evaluation" },
  { key: "registry", label: "Registry" },
  { key: "deployment", label: "Deployment" },
  { key: "runtime_health", label: "Runtime Health" },
];

const STEPS: Array<{ key: TabKey; label: string; stateKey: keyof WorkflowState }> = [
  { key: "datasets", label: "Dataset", stateKey: "dataset_ready" },
  { key: "labels", label: "Labels", stateKey: "labels_ready" },
  { key: "features", label: "Features", stateKey: "feature_export_ready" },
  { key: "experiments", label: "Train", stateKey: "training_complete" },
  { key: "evaluation", label: "Evaluate", stateKey: "evaluation_ready" },
  { key: "registry", label: "Promote", stateKey: "promotion_eligible" },
];

const DEFAULT_DATASET: MLDatasetSummary = {
  id: "ml_dataset_1",
  name: "ML Dataset 1",
  source_dataset_ids: [],
  filters: { exclude_archived: true },
  label_schema: { task: "classification" },
  object_count: 0,
  tags: [],
  notes: null,
};

type WorkflowState = {
  dataset_ready: boolean;
  labels_ready: boolean;
  leakage_risk: boolean;
  feature_export_ready: boolean;
  training_complete: boolean;
  evaluation_ready: boolean;
  promotion_eligible: boolean;
  warnings: string[];
};

function Badge({ label, tone }: { label: string; tone: BadgeTone }) {
  return <span className={`ml-badge ml-badge-${tone}`}>{label}</span>;
}

function toneForDeployment(dep: MLDeploymentSummary): BadgeTone {
  if (dep.status === "superseded") return "superseded";
  if (dep.status === "invalid") return "invalid";
  if (dep.status === "active") {
    const compat = Boolean((dep.runtime_validation as Record<string, unknown> | undefined)?.compatible);
    return compat ? "active" : "incompatible";
  }
  return "heuristic";
}

function statusTone(passed: boolean | null): BadgeTone {
  if (passed === null) return "warning";
  return passed ? "healthy" : "failed";
}

function compactNumber(value: unknown): string {
  const n = Number(value ?? 0);
  return Number.isFinite(n) ? n.toLocaleString() : "-";
}

export default function ClassifiersPage() {
  const [tab, setTab] = useState<TabKey>("overview");

  const [datasets, setDatasets] = useState<MLDatasetSummary[]>([]);
  const [experiments, setExperiments] = useState<MLExperimentSummary[]>([]);
  const [backends, setBackends] = useState<MLBackendInfo[]>([]);
  const [models, setModels] = useState<Array<Record<string, unknown>>>([]);
  const [deployments, setDeployments] = useState<MLDeploymentSummary[]>([]);

  const [smokeLatest, setSmokeLatest] = useState<MLSmokeSummary | null>(null);
  const [runtimeHealth, setRuntimeHealth] = useState<any>(null);
  const [runtimeHistory, setRuntimeHistory] = useState<Array<{ day: string; stats: Record<string, unknown> }>>([]);

  const [datasetId, setDatasetId] = useState<string>("");
  const [datasetInsights, setDatasetInsights] = useState<MLDatasetInsights | null>(null);
  const [labelsPreview, setLabelsPreview] = useState<{ count: number; items: Array<Record<string, unknown>> } | null>(null);
  const [featuresPreview, setFeaturesPreview] = useState<{
    count: number;
    schema: Record<string, unknown>;
    diagnostics: Record<string, unknown>;
    rows: Array<Record<string, unknown>>;
  } | null>(null);
  const [splitPreview, setSplitPreview] = useState<Record<string, unknown> | null>(null);
  const [evaluationPreview, setEvaluationPreview] = useState<Record<string, unknown> | null>(null);
  const [trainingLogs, setTrainingLogs] = useState<Record<string, unknown> | null>(null);
  const [selectedExperiment, setSelectedExperiment] = useState<string>("");

  const [globalError, setGlobalError] = useState<string | null>(null);
  const [overviewLoading, setOverviewLoading] = useState(false);
  const [datasetLoading, setDatasetLoading] = useState(false);
  const [smokeBusy, setSmokeBusy] = useState(false);
  const [splitStrategy, setSplitStrategy] = useState("by_session");
  const [compareMode, setCompareMode] = useState(false);
  const [selectedBackends, setSelectedBackends] = useState<string[]>(["logistic_regression"]);
  const [draft, setDraft] = useState<MLDatasetSummary>(DEFAULT_DATASET);

  const isDevMode = Boolean((import.meta as ImportMeta).env?.DEV);

  async function load() {
    setGlobalError(null);
    setOverviewLoading(true);
    const jobs = await Promise.allSettled([
      api.mlDatasets(),
      api.mlExperiments(),
      api.mlBackends(),
      api.mlModels(),
      api.mlDeployments(),
      api.mlSmokeLatest(),
      api.mlRuntimeHealth(),
      api.mlRuntimeHealthHistory(),
    ]);
    setOverviewLoading(false);

    const errors: string[] = [];
    if (jobs[0].status === "fulfilled") {
      setDatasets(jobs[0].value);
      if (!datasetId && jobs[0].value[0]?.id) setDatasetId(jobs[0].value[0].id);
    } else errors.push("datasets");

    if (jobs[1].status === "fulfilled") {
      setExperiments(jobs[1].value);
      if (!selectedExperiment && jobs[1].value[0]?.id) setSelectedExperiment(jobs[1].value[0].id);
    } else errors.push("experiments");

    if (jobs[2].status === "fulfilled") setBackends(jobs[2].value);
    else errors.push("backends");

    if (jobs[3].status === "fulfilled") setModels(jobs[3].value);
    else errors.push("models");

    if (jobs[4].status === "fulfilled") setDeployments(jobs[4].value);
    else errors.push("deployments");

    setSmokeLatest(jobs[5].status === "fulfilled" ? jobs[5].value : null);
    if (jobs[5].status !== "fulfilled") errors.push("smoke");

    setRuntimeHealth(jobs[6].status === "fulfilled" ? jobs[6].value : null);
    if (jobs[6].status !== "fulfilled") errors.push("runtime health");

    setRuntimeHistory(jobs[7].status === "fulfilled" ? (jobs[7].value.items ?? []) : []);
    if (jobs[7].status !== "fulfilled") errors.push("runtime history");

    if (errors.length > 0) setGlobalError(`Some panels are degraded: ${errors.join(", ")}`);
  }

  useEffect(() => {
    void load();
  }, []);

  useEffect(() => {
    if (!datasetId) return;
    void (async () => {
      setDatasetLoading(true);
      const jobs = await Promise.allSettled([
        api.mlDatasetInsights(datasetId),
        api.mlLabelsPreview(datasetId, 60),
        api.mlFeaturesPreview(datasetId, 40),
        api.mlSplitPreview(datasetId, splitStrategy, 42),
      ]);
      setDatasetLoading(false);

      setDatasetInsights(jobs[0].status === "fulfilled" ? jobs[0].value : null);
      setLabelsPreview(jobs[1].status === "fulfilled" ? { count: jobs[1].value.count, items: jobs[1].value.items } : null);
      setFeaturesPreview(jobs[2].status === "fulfilled" ? jobs[2].value : null);
      setSplitPreview(jobs[3].status === "fulfilled" ? jobs[3].value : null);
    })();
  }, [datasetId, splitStrategy]);

  useEffect(() => {
    if (!selectedExperiment) return;
    void (async () => {
      const jobs = await Promise.allSettled([
        api.mlExperimentLogs(selectedExperiment),
        api.mlEvaluationPreview(selectedExperiment),
      ]);
      setTrainingLogs(jobs[0].status === "fulfilled" ? (jobs[0].value as unknown as Record<string, unknown>) : null);
      setEvaluationPreview(jobs[1].status === "fulfilled" ? (jobs[1].value as unknown as Record<string, unknown>) : null);
    })();
  }, [selectedExperiment]);

  const workflowState = useMemo<WorkflowState>(() => {
    const warnings = datasetInsights?.warnings ?? [];
    const insufficient = warnings.includes("insufficient_class_diversity") || warnings.includes("low_validation_coverage");
    const leakageRisk =
      Array.isArray((splitPreview as any)?.leakage_report?.warnings) &&
      ((splitPreview as any).leakage_report.warnings as unknown[]).length > 0;
    const trainingComplete = experiments.some((e) => e.status === "completed");
    const evaluationReady = experiments.some((e) => Boolean((e.metrics_summary ?? {}).accuracy !== undefined));
    return {
      dataset_ready: (datasetInsights?.object_count ?? 0) > 0,
      labels_ready: !insufficient,
      leakage_risk: leakageRisk,
      feature_export_ready: (featuresPreview?.count ?? 0) > 0,
      training_complete: trainingComplete,
      evaluation_ready: evaluationReady,
      promotion_eligible: models.some((m) => Boolean(m.promoted)) || trainingComplete,
      warnings,
    };
  }, [datasetInsights, splitPreview, featuresPreview, experiments, models]);

  const smokePassed = smokeLatest ? Boolean(smokeLatest.passed) : null;
  const smokeFailures = (smokeLatest?.failures ?? []).length;

  return (
    <main className="classifiers-page">
      <section className="classifiers-shell">
        <header className="classifiers-header-panel">
          <div>
            <h2>Classifiers</h2>
            <p>Guided experimentation workflow for mining/25D with safe deployment controls.</p>
          </div>
          <Badge label={smokePassed === null ? "WARNING" : smokePassed ? "HEALTHY" : "FAILED"} tone={statusTone(smokePassed)} />
        </header>

        <section className="classifiers-stepper" aria-label="Guided ML workflow">
          {STEPS.map((step) => {
            const value = workflowState[step.stateKey];
            const tone: BadgeTone = value ? "healthy" : step.stateKey === "labels_ready" || step.stateKey === "evaluation_ready" ? "warning" : "neutral";
            if (step.stateKey === "labels_ready" && workflowState.warnings.length > 0) {
              return (
                <button key={step.key} type="button" className="classifiers-step classifiers-step-warning" onClick={() => setTab(step.key)}>
                  <span>{step.label}</span>
                  <Badge label="WARNING" tone="warning" />
                </button>
              );
            }
            if (step.stateKey === "training_complete" && workflowState.leakage_risk) {
              return (
                <button key={step.key} type="button" className="classifiers-step classifiers-step-warning" onClick={() => setTab(step.key)}>
                  <span>{step.label}</span>
                  <Badge label="LEAKAGE RISK" tone="warning" />
                </button>
              );
            }
            return (
              <button key={step.key} type="button" className={`classifiers-step ${tab === step.key ? "active" : ""}`} onClick={() => setTab(step.key)}>
                <span>{step.label}</span>
                <Badge label={value ? "READY" : "PENDING"} tone={tone} />
              </button>
            );
          })}
        </section>

        <nav className="classifiers-tabs" aria-label="Classifiers sections">
          {TABS.map((t) => (
            <button key={t.key} type="button" className={tab === t.key ? "active" : ""} onClick={() => setTab(t.key)}>
              {t.label}
            </button>
          ))}
        </nav>

        {globalError && (
          <div className="ml-alert ml-alert-warning">
            <strong>Partial data available.</strong>
            <span>{globalError}</span>
          </div>
        )}

        {(tab === "overview" || tab === "runtime_health") && (
          <section className="ml-panel ml-panel-dense">
            <div className="ml-panel-head">
              <h3>ML System Health</h3>
              {overviewLoading && <span className="ml-muted">Refreshing...</span>}
            </div>

            <div className="ml-kpi-grid ml-kpi-strip">
              <article className="ml-kpi-card"><small>Inference Count</small><strong>{compactNumber(runtimeHealth?.totals?.inference_count ?? 0)}</strong></article>
              <article className="ml-kpi-card"><small>Fallback Rate</small><strong>{compactNumber(((runtimeHealth?.totals?.fallback_rate ?? 0) * 100).toFixed(2))}%</strong></article>
              <article className="ml-kpi-card"><small>Inference p95</small><strong>{runtimeHealth?.totals?.p95_inference_time_ms == null ? "-" : `${compactNumber(Number(runtimeHealth.totals.p95_inference_time_ms).toFixed(1))} ms`}</strong></article>
              <article className="ml-kpi-card"><small>Heuristic Usage</small><strong>{compactNumber(((runtimeHealth?.totals?.heuristic_usage_rate ?? 0) * 100).toFixed(2))}%</strong></article>
              <article className="ml-kpi-card"><small>Avg Inference Time</small><strong>{compactNumber(runtimeHealth?.totals?.average_inference_time_ms ?? 0)} ms</strong></article>
              <article className="ml-kpi-card"><small>Incompatible Deployments</small><strong>{compactNumber(runtimeHealth?.totals?.incompatible_deployment_count ?? 0)}</strong></article>
            </div>

            <div className="ml-workspace-grid ml-workspace-grid-health">
              <article className="ml-card ml-card-compact">
                <div className="ml-card-head">
                  <h4>Latest Smoke Test</h4>
                  <Badge label={smokePassed === null ? "WARNING" : smokePassed ? "PASS" : "FAIL"} tone={statusTone(smokePassed)} />
                </div>
                {!smokeLatest ? (
                  <p className="ml-empty">No smoke result available yet.</p>
                ) : (
                  <div className="ml-list-grid">
                    <span>Timestamp</span><strong>{smokeLatest.timestamp ?? "-"}</strong>
                    <span>Duration</span><strong>{compactNumber(smokeLatest.duration_ms ?? 0)} ms</strong>
                    <span>Fallback Validation</span><strong>{String(Boolean(smokeLatest.fallback_mode))}</strong>
                    <span>Failure Count</span><strong>{compactNumber(smokeFailures)}</strong>
                  </div>
                )}
                {isDevMode && (
                  <div className="ml-actions-row">
                    <button
                      type="button"
                      disabled={smokeBusy}
                      className="ml-btn"
                      onClick={async () => {
                        try {
                          setSmokeBusy(true);
                          await api.runMlSmokeTest({ ci_mode: true });
                          await load();
                        } finally {
                          setSmokeBusy(false);
                        }
                      }}
                    >
                      {smokeBusy ? "Running..." : "Run smoke test"}
                    </button>
                    <button
                      type="button"
                      disabled={smokeBusy}
                      className="ml-btn ml-btn-subtle"
                      onClick={async () => {
                        try {
                          setSmokeBusy(true);
                          await api.runMlSmokeTest({ ci_mode: true, force_invalid_model: true });
                          await load();
                        } finally {
                          setSmokeBusy(false);
                        }
                      }}
                    >
                      {smokeBusy ? "Running..." : "Run fallback smoke test"}
                    </button>
                  </div>
                )}
              </article>

              {tab === "runtime_health" && (
                <article className="ml-card ml-card-compact">
                  <div className="ml-card-head"><h4>Runtime History</h4></div>
                  {!runtimeHistory.length ? (
                    <p className="ml-empty">No runtime history yet.</p>
                  ) : (
                    <div className="ml-table-wrap">
                      <table className="ml-table compact">
                        <thead>
                          <tr>
                            <th>Date</th>
                            <th>Inference</th>
                            <th>Fallback</th>
                            <th>Avg ms</th>
                          </tr>
                        </thead>
                        <tbody>
                          {runtimeHistory
                            .slice(0, 14)
                            .sort((a, b) => String(b.day).localeCompare(String(a.day)))
                            .map((row) => (
                              <tr key={row.day}>
                                <td><code>{row.day}</code></td>
                                <td>{compactNumber((row.stats as any).inference_count ?? 0)}</td>
                                <td>{compactNumber((row.stats as any).fallback_count ?? 0)}</td>
                                <td>{compactNumber((row.stats as any).average_inference_time_ms ?? 0)}</td>
                              </tr>
                            ))}
                        </tbody>
                      </table>
                    </div>
                  )}
                </article>
              )}
            </div>
          </section>
        )}

        {(tab === "overview" || tab === "datasets") && (
          <section className="ml-panel">
            <div className="ml-panel-head"><h3>Datasets</h3></div>
            <div className="ml-actions-row">
              <input value={draft.id} onChange={(e) => setDraft((v) => ({ ...v, id: e.target.value }))} placeholder="dataset id" />
              <input value={draft.name} onChange={(e) => setDraft((v) => ({ ...v, name: e.target.value }))} placeholder="dataset name" />
              <button type="button" className="ml-btn" onClick={async () => { await api.createMlDataset(draft); await load(); }}>Create dataset</button>
            </div>

            {!datasets.length ? (
              <div className="ml-empty-state">
                <h4>No ML datasets yet</h4>
                <p>Create a dataset to start the guided workflow.</p>
              </div>
            ) : (
              <>
                <div className="ml-actions-row">
                  <label>
                    Active dataset
                    <select value={datasetId} onChange={(e) => setDatasetId(e.target.value)}>
                      {datasets.map((d) => <option key={d.id} value={d.id}>{d.id}</option>)}
                    </select>
                  </label>
                  <label>
                    Split strategy
                    <select value={splitStrategy} onChange={(e) => setSplitStrategy(e.target.value)}>
                      <option value="random">random</option>
                      <option value="stratified">stratified</option>
                      <option value="by_session">by_session</option>
                      <option value="by_dataset">by_dataset</option>
                      <option value="chronological">chronological</option>
                    </select>
                  </label>
                </div>

                {datasetLoading && <p className="ml-muted">Loading dataset insights...</p>}

                {datasetInsights && (
                  <div className="ml-grid-2">
                    <article className="ml-card">
                      <div className="ml-card-head"><h4>Dataset Quality</h4></div>
                      <div className="ml-list-grid">
                        <span>Objects</span><strong>{compactNumber(datasetInsights.object_count)}</strong>
                        <span>Class Balance</span><strong>{Object.entries(datasetInsights.class_counts).map(([k, v]) => `${k}:${v}`).join(" | ") || "-"}</strong>
                        <span>Session Distribution</span><strong>{Object.entries(datasetInsights.session_counts).map(([k, v]) => `${k}:${v}`).join(" | ") || "-"}</strong>
                        <span>Validation Coverage</span><strong>{Object.entries(datasetInsights.validation_counts).map(([k, v]) => `${k}:${v}`).join(" | ") || "-"}</strong>
                      </div>
                    </article>
                    <article className="ml-card">
                      <div className="ml-card-head"><h4>Warnings</h4></div>
                      {datasetInsights.warnings.length === 0 ? (
                        <Badge label="HEALTHY" tone="healthy" />
                      ) : (
                        <div className="ml-alert-stack">
                          {datasetInsights.warnings.map((warning) => (
                            <div key={warning} className="ml-alert ml-alert-warning"><span>⚠</span><strong>{warning}</strong></div>
                          ))}
                        </div>
                      )}
                    </article>
                  </div>
                )}
              </>
            )}
          </section>
        )}

        {tab === "labels" && (
          <section className="ml-panel">
            <div className="ml-panel-head"><h3>Labels</h3></div>
            {!labelsPreview ? (
              <p className="ml-empty">No labels preview available.</p>
            ) : (
              <>
                <p className="ml-muted">Labeled objects: {compactNumber(labelsPreview.count)}</p>
                <div className="ml-table-wrap">
                  <table className="ml-table compact">
                    <thead>
                      <tr><th>Take</th><th>Object</th><th>Class</th><th>Validation</th></tr>
                    </thead>
                    <tbody>
                      {labelsPreview.items.slice(0, 60).map((item, idx) => (
                        <tr key={`${String(item.take_id)}_${String(item.object_id)}_${idx}`}>
                          <td><code>{String(item.take_id)}</code></td>
                          <td>{String(item.object_id)}</td>
                          <td>{String(item.expected_class)}</td>
                          <td>{String(item.validation_status ?? "-")}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </>
            )}
          </section>
        )}

        {tab === "features" && (
          <section className="ml-panel">
            <div className="ml-panel-head"><h3>Features</h3></div>
            {!featuresPreview ? (
              <p className="ml-empty">No features preview available.</p>
            ) : (
              <>
                <div className="ml-grid-2">
                  <article className="ml-card">
                    <h4>Schema</h4>
                    <div className="ml-list-grid">
                      <span>Rows</span><strong>{compactNumber(featuresPreview.count)}</strong>
                      <span>Feature Count</span><strong>{String(featuresPreview.schema.feature_count ?? "-")}</strong>
                      <span>Ordering Hash</span><strong><code>{String(featuresPreview.schema.feature_ordering_hash ?? "-")}</code></strong>
                      <span>Fingerprint</span><strong><code>{String(featuresPreview.schema.schema_fingerprint ?? "-")}</code></strong>
                    </div>
                  </article>
                  <article className="ml-card">
                    <h4>Diagnostics</h4>
                    <div className="ml-alert-stack">
                      <div className="ml-alert ml-alert-info"><span>i</span><strong>Constant: {JSON.stringify(featuresPreview.diagnostics.constant_features ?? [])}</strong></div>
                      <div className="ml-alert ml-alert-info"><span>i</span><strong>Sparse: {JSON.stringify(featuresPreview.diagnostics.sparse_features ?? [])}</strong></div>
                    </div>
                  </article>
                </div>
                <div className="ml-table-wrap">
                  <table className="ml-table compact">
                    <thead>
                      <tr><th>Take</th><th>Object</th><th>Expected</th></tr>
                    </thead>
                    <tbody>
                      {featuresPreview.rows.slice(0, 40).map((row, idx) => (
                        <tr key={`${String(row.take_id)}_${String(row.object_id)}_${idx}`}>
                          <td><code>{String(row.take_id)}</code></td>
                          <td>{String(row.object_id)}</td>
                          <td>{String(row.expected_class)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </>
            )}
          </section>
        )}

        {(tab === "overview" || tab === "experiments") && (
          <section className="ml-panel ml-panel-dense">
            <div className="ml-panel-head"><h3>Experiments</h3></div>
            <div className="ml-workspace-grid ml-workspace-grid-experiments">
              <article className="ml-card ml-card-compact">
                <h4>Training Wizard (Compact)</h4>
                <div className="ml-list-grid">
                  <span>Dataset Ready</span><strong>{String(workflowState.dataset_ready)}</strong>
                  <span>Leakage Risk</span><strong>{String(workflowState.leakage_risk)}</strong>
                  <span>Feature Export Ready</span><strong>{String(workflowState.feature_export_ready)}</strong>
                  <span>Promotion Eligible</span><strong>{String(workflowState.promotion_eligible)}</strong>
                </div>
                {workflowState.leakage_risk && (
                  <div className="ml-alert ml-alert-warning"><span>⚠</span><strong>Leakage warning detected. Confirm before training.</strong></div>
                )}
                <div className="ml-actions-row">
                  <label>
                    Mode
                    <select value={compareMode ? "comparison" : "single"} onChange={(e) => setCompareMode(e.target.value === "comparison")}>
                      <option value="single">Single backend</option>
                      <option value="comparison">Compare multiple classifiers</option>
                    </select>
                  </label>
                </div>
                <div className="ml-backend-grid">
                  {backends.map((backend) => {
                    const backendId = String(backend.id);
                    const selected = selectedBackends.includes(backendId);
                    const disabled = backend.enabled === false;
                    return (
                      <button
                        key={backendId}
                        type="button"
                        disabled={disabled}
                        className={`ml-backend-chip ${selected ? "active" : ""}`}
                        onClick={() => {
                          if (compareMode) {
                            setSelectedBackends((prev) => selected ? prev.filter((v) => v !== backendId) : [...prev, backendId]);
                          } else {
                            setSelectedBackends([backendId]);
                          }
                        }}
                      >
                        <strong>{backendId}</strong>
                        <small>{backend.description ?? "-"}</small>
                        <small>interp={backend.interpretability_level} | train={backend.training_speed} | infer={backend.inference_speed}</small>
                      </button>
                    );
                  })}
                </div>
                <div className="ml-actions-row">
                  <button
                    type="button"
                    className="ml-btn"
                    onClick={async () => {
                      if (!datasetId) return;
                      if (workflowState.leakage_risk) {
                        const ok = window.confirm("Leakage risk warnings were detected for the current split. Continue training?");
                        if (!ok) return;
                      }
                      await api.runMlExperiment({
                        id: `exp_${Date.now()}`,
                        name: compareMode ? "Classifier comparison" : "Guided experiment",
                        classifier_type: selectedBackends[0] || "logistic_regression",
                        classifier_types: compareMode ? selectedBackends : [],
                        experiment_mode: compareMode ? "comparison_experiment" : "single",
                        feature_set_id: "core_25d_v1",
                        dataset_id: datasetId,
                        split_strategy: splitStrategy,
                        training_config: { seed: 42 },
                      });
                      await load();
                    }}
                  >
                    Run training
                  </button>
                </div>
              </article>

              <article className="ml-card ml-card-compact">
                <h4>Split and Leakage</h4>
                <details>
                  <summary>Review split manifest and leakage diagnostics</summary>
                  <pre className="ml-code-block">{JSON.stringify(splitPreview ?? {}, null, 2)}</pre>
                </details>
              </article>
            </div>

            <div className="ml-actions-row">
              <label>
                Selected experiment
                <select value={selectedExperiment} onChange={(e) => setSelectedExperiment(e.target.value)}>
                  {experiments.map((e) => <option key={e.id} value={e.id}>{e.id}</option>)}
                </select>
              </label>
            </div>

            {!experiments.length ? (
              <div className="ml-empty-state"><h4>No experiments yet</h4><p>Run a training job to generate evaluation artifacts and models.</p></div>
            ) : (
              <div className="ml-table-wrap">
                <table className="ml-table compact">
                  <thead><tr><th>Id</th><th>Status</th><th>Classifier</th><th>Split</th></tr></thead>
                  <tbody>
                    {experiments.slice(0, 50).map((exp) => (
                      <tr key={exp.id}>
                        <td><code>{exp.id}</code></td>
                        <td><Badge label={exp.status.toUpperCase()} tone={exp.status === "completed" ? "healthy" : exp.status === "failed" ? "failed" : "warning"} /></td>
                        <td>{exp.classifier_types?.length ? exp.classifier_types.join(", ") : exp.classifier_type}</td>
                        <td>{exp.split_strategy}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}

            <details>
              <summary>Training logs</summary>
              <pre className="ml-code-block">{JSON.stringify(trainingLogs ?? {}, null, 2)}</pre>
            </details>
          </section>
        )}

        {tab === "evaluation" && (
          <section className="ml-panel">
            <div className="ml-panel-head"><h3>Evaluation Workspace</h3></div>
            {!evaluationPreview ? (
              <p className="ml-empty">No evaluation data available.</p>
            ) : (
              <div className="ml-grid-2">
                <article className="ml-card">
                  <h4>Summary</h4>
                  <pre className="ml-code-block">{JSON.stringify((evaluationPreview as any).summary ?? {}, null, 2)}</pre>
                </article>
                <article className="ml-card">
                  <h4>Failed Examples</h4>
                  <p className="ml-muted">Count: {compactNumber(((evaluationPreview as any).failed_examples ?? []).length)}</p>
                  <details>
                    <summary>Open failed examples details</summary>
                    <pre className="ml-code-block">{JSON.stringify((evaluationPreview as any).failed_examples ?? [], null, 2)}</pre>
                  </details>
                </article>
              </div>
            )}
            {(() => {
              const comp = (evaluationPreview as any)?.comparison_summary ?? {};
              const ranking = Array.isArray(comp.ranking) ? comp.ranking : [];
              if (!ranking.length) return null;
              return (
                <article className="ml-card">
                  <div className="ml-card-head">
                    <h4>Backend Comparison</h4>
                    <Badge label={`RECOMMENDED: ${String(comp.best_backend_candidate ?? "-")}`} tone="healthy" />
                  </div>
                  <div className="ml-table-wrap">
                    <table className="ml-table compact">
                      <thead><tr><th>Backend</th><th>F1</th><th>Precision</th><th>Recall</th><th>Train ms</th><th>Infer ms</th></tr></thead>
                      <tbody>
                        {ranking.slice(0, 8).map((row: any, idx: number) => (
                          <tr key={`${String(row.backend_id)}_${idx}`}>
                            <td>{String(row.backend_id)}</td>
                            <td>{Number(row.f1_macro ?? 0).toFixed(3)}</td>
                            <td>{Number(row.precision_macro ?? 0).toFixed(3)}</td>
                            <td>{Number(row.recall_macro ?? 0).toFixed(3)}</td>
                            <td>{compactNumber((row.training_duration_ms ?? 0).toFixed?.(2) ?? row.training_duration_ms ?? 0)}</td>
                            <td>{compactNumber((row.inference_duration_ms ?? 0).toFixed?.(2) ?? row.inference_duration_ms ?? 0)}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </article>
              );
            })()}
          </section>
        )}

        {tab === "registry" && (
          <section className="ml-panel">
            <div className="ml-panel-head"><h3>Registry</h3></div>
            {!models.length ? (
              <div className="ml-empty-state"><h4>No trained models</h4><p>Train an experiment to populate the registry.</p></div>
            ) : (
              <div className="ml-table-wrap">
                <table className="ml-table compact">
                  <thead><tr><th>Model</th><th>Promoted</th><th>Backend</th><th>Action</th></tr></thead>
                  <tbody>
                    {models.slice(0, 50).map((m, idx) => (
                      <tr key={`${String(m.id)}_${idx}`}>
                        <td><code>{String(m.id ?? "-")}</code></td>
                        <td>{Boolean(m.promoted) ? <Badge label="HEALTHY" tone="healthy" /> : <Badge label="WARNING" tone="warning" />}</td>
                        <td>{String((m.backend as any)?.backend_name ?? "-")}</td>
                        <td>
                          <button
                            type="button"
                            className="ml-btn ml-btn-subtle"
                            onClick={async () => {
                              const ok = window.confirm(`Promote model ${String(m.id)}?`);
                              if (!ok) return;
                              await api.promoteMlModel(String(m.id));
                              await load();
                            }}
                          >
                            Promote
                          </button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </section>
        )}

        {tab === "deployment" && (
          <section className="ml-panel ml-panel-dense">
            <div className="ml-panel-head"><h3>Deployment</h3></div>

            {!models.length ? (
              <div className="ml-empty-state"><h4>No models available</h4><p>Train and promote at least one model before deployment.</p></div>
            ) : (
              <div className="ml-actions-row">
                <button
                  type="button"
                  className="ml-btn"
                  onClick={async () => {
                    const first = models[0];
                    const modelId = String(first?.id ?? "");
                    if (!modelId) return;
                    const ok = window.confirm(
                      `Create deployment for model ${modelId} targeting pipeline 25d_ball_inspection/classification (family 25d)?`
                    );
                    if (!ok) return;
                    await api.createMlDeployment({
                      id: `dep_${Date.now()}`,
                      model_id: modelId,
                      target_pipeline_id: "25d_ball_inspection",
                      target_stage_id: "classification",
                      pipeline_family: "25d",
                      notes: "Created from guided UI",
                    });
                    await load();
                  }}
                >
                  Create deployment
                </button>
              </div>
            )}

            {!deployments.length ? (
              <div className="ml-empty-state"><h4>No deployments</h4><p>Create and activate a deployment to route runtime inference to ML models.</p></div>
            ) : (
              <div className="ml-workspace-grid ml-workspace-grid-deployment">
                <div className="ml-table-wrap">
                  <table className="ml-table compact">
                    <thead>
                      <tr>
                        <th>Deployment</th>
                        <th>Model</th>
                        <th>Target</th>
                        <th>Status</th>
                        <th>Lineage</th>
                        <th>Actions</th>
                      </tr>
                    </thead>
                    <tbody>
                      {deployments.slice(0, 60).map((dep) => (
                        <tr key={dep.id} className={dep.status === "active" ? "ml-row-active" : ""}>
                          <td><code>{dep.id}</code></td>
                          <td><code>{dep.model_id}</code></td>
                          <td>{dep.target_pipeline_id}/{dep.target_stage_id}/{dep.pipeline_family}</td>
                          <td><Badge label={dep.status === "invalid" ? "INVALID MODEL" : dep.status.toUpperCase()} tone={toneForDeployment(dep)} /></td>
                          <td><code>rollback_from={String(dep.rollback_from ?? "-")}</code></td>
                          <td>
                            <div className="ml-actions-row">
                              {dep.status === "active" && <button type="button" className="ml-btn" onClick={async () => { const result = await api.startMlLiveTrial({ deployment_id: dep.id, notes: "Started from deployment workspace" }); window.alert(`Live trial frozen: ${result.id}`); }}>Start live trial</button>}
                              <button
                                type="button"
                                className="ml-btn"
                                onClick={async () => {
                                  const ok = window.confirm(
                                    `Activate ${dep.id}? This may supersede the current active deployment for ${dep.target_pipeline_id}/${dep.target_stage_id}/${dep.pipeline_family}.`
                                  );
                                  if (!ok) return;
                                  await api.activateMlDeployment(dep.id);
                                  await load();
                                }}
                              >
                                Activate
                              </button>
                              <button
                                type="button"
                                className="ml-btn ml-btn-subtle"
                                onClick={async () => {
                                  const ok = window.confirm(`Deactivate ${dep.id}?`);
                                  if (!ok) return;
                                  await api.deactivateMlDeployment(dep.id);
                                  await load();
                                }}
                              >
                                Deactivate
                              </button>
                              <button
                                type="button"
                                className="ml-btn ml-btn-danger"
                                onClick={async () => {
                                  const ok = window.confirm(`Rollback from ${dep.id}? This creates explicit rollback lineage.`);
                                  if (!ok) return;
                                  await api.rollbackMlDeployment(dep.id);
                                  await load();
                                }}
                              >
                                Rollback
                              </button>
                            </div>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                <div className="ml-card ml-card-compact">
                  <h4>Deployment Diagnostics</h4>
                  {deployments.slice(0, 8).map((dep) => (
                    <details key={`dep_diag_${dep.id}`} className="ml-diagnostic-summary">
                      <summary>
                        <code>{dep.id}</code>
                        <Badge label={dep.status.toUpperCase()} tone={toneForDeployment(dep)} />
                      </summary>
                      <pre className="ml-code-block">{JSON.stringify({
                        model_id: dep.model_id,
                        target: `${dep.target_pipeline_id}/${dep.target_stage_id}/${dep.pipeline_family}`,
                        compatibility: dep.compatibility_snapshot,
                        runtime: dep.runtime_validation,
                      }, null, 2)}</pre>
                    </details>
                  ))}
                </div>
              </div>
            )}
          </section>
        )}
      </section>
    </main>
  );
}
