import type { FeatureAnalyticsDistributionResponse, FeatureAnalyticsFeaturesResponse, PipelineInfo } from "./api/client";

export type FeatureFreshnessScope = {
  scope: "ml_set" | "dataset" | "takes";
  datasetId?: string;
  mlSetId?: string;
  takeIds?: string[];
  takeCount: number;
  mlSetName?: string | null;
};

export type FeatureFreshnessState = "complete" | "all_missing" | "partial_missing" | "not_applicable";

export function featureFreshnessState(args: {
  featureKey: string;
  featureRegistered: boolean;
  scopedObjectCount: number;
  distribution: FeatureAnalyticsDistributionResponse | null;
}): FeatureFreshnessState {
  const { featureKey, featureRegistered, scopedObjectCount, distribution } = args;
  if (!featureKey || !featureRegistered || scopedObjectCount <= 0) return "not_applicable";
  const count = distribution?.stats.count ?? 0;
  const missingPct = distribution?.stats.missing_pct ?? 0;
  if (count === 0 && missingPct >= 100) return "all_missing";
  if (missingPct > 0 && missingPct < 100) return "partial_missing";
  if (count > 0 && missingPct === 0) return "complete";
  if (count > 0 && missingPct > 0) return "partial_missing";
  return "not_applicable";
}

export function resolveFeatureFreshnessScope(args: {
  datasetId: string;
  mlSetId: string;
  takeIds: string[];
  mlSetName?: string | null;
  scopeSummary?: FeatureAnalyticsFeaturesResponse["scope_summary"] | null;
}): FeatureFreshnessScope | null {
  const takeCount = args.scopeSummary?.take_count ?? (args.takeIds.length || 0);
  if (args.mlSetId) {
    return {
      scope: "ml_set",
      datasetId: args.datasetId || undefined,
      mlSetId: args.mlSetId,
      takeCount: takeCount || 0,
      mlSetName: args.mlSetName,
    };
  }
  if (args.takeIds.length) {
    return {
      scope: "takes",
      datasetId: args.datasetId || undefined,
      takeIds: args.takeIds,
      takeCount: args.takeIds.length,
    };
  }
  if (args.datasetId) {
    return {
      scope: "dataset",
      datasetId: args.datasetId,
      takeCount: takeCount || 0,
    };
  }
  return null;
}

export function defaultFeaturePipelineId(pipelines: PipelineInfo[]): string | null {
  const preferred = pipelines.find((item) => item.id === "mining_steel_ball_classification_25d");
  if (preferred) return preferred.id;
  const ball = pipelines.find((item) => item.id.includes("25d"));
  return ball?.id ?? pipelines[0]?.id ?? null;
}

const FEATURE_JOB_BACKFILL_KEYS = new Set(["surface_sphere_fit_rmse_mm"]);

export function featureJobActionsEnabled(args: {
  featureKey: string;
  featureRegistered: boolean;
  freshness: FeatureFreshnessState;
  scope: FeatureFreshnessScope | null;
  pipelineId: string | null;
  diagnosticOnly?: boolean;
  featureFamily?: string | null;
}): { reprocess: boolean; backfill: boolean; reason?: string } {
  if (!args.featureKey || !args.featureRegistered) {
    return { reprocess: false, backfill: false, reason: "Feature is not registered for compute jobs." };
  }
  if (!args.scope || args.scope.takeCount <= 0) {
    return { reprocess: false, backfill: false, reason: "Select an ML set, dataset, or explicit take scope." };
  }
  if (args.freshness === "complete") {
    return { reprocess: false, backfill: false, reason: "Feature values are complete in the current scope." };
  }
  if (args.freshness === "not_applicable") {
    return { reprocess: false, backfill: false, reason: "No objects in the current scope." };
  }
  const sphereConsistency = String(args.featureFamily ?? "") === "sphere_consistency";
  const reprocess = Boolean(args.pipelineId) && sphereConsistency;
  const backfill = FEATURE_JOB_BACKFILL_KEYS.has(args.featureKey);
  let reason: string | undefined;
  if (!sphereConsistency) {
    reason = "Compute jobs are currently supported for sphere-consistency features only.";
  } else if (!args.pipelineId) {
    reason = "No compatible pipeline resolved for reprocess.";
  } else if (!backfill) {
    reason = "Backfill is only available for raw sphere-fit RMSE. Use reprocess for normalized sphere-consistency features.";
  }
  return {
    reprocess,
    backfill,
    reason: reprocess || backfill ? undefined : reason,
  };
}

export function buildFeatureJobCliCommand(args: {
  mode: "reprocess" | "backfill";
  featureKey: string;
  scope: FeatureFreshnessScope;
  pipelineId?: string | null;
}): string {
  const base = "python scripts/sensor_studio_cli.py --data-dir data ml compute-feature";
  const mlSet = args.scope.mlSetId ? ` --ml-set-id ${args.scope.mlSetId}` : "";
  const dataset = args.scope.datasetId ? ` --dataset-id ${args.scope.datasetId}` : "";
  const pipeline = args.mode === "reprocess" && args.pipelineId ? ` --pipeline-id ${args.pipelineId}` : "";
  return `${base}${dataset}${mlSet} --feature ${args.featureKey} --mode ${args.mode} --apply${pipeline}`;
}

export function featureSourceLabel(source: string | null | undefined): string {
  if (source === "feature_backfill") return "backfill";
  if (source === "derived") return "derived";
  if (source === "pipeline_run" || source === "pipeline_reprocess") return "pipeline";
  return source ? String(source) : "-";
}

export function featureSourceTooltip(source: string | null | undefined): string {
  if (source === "feature_backfill") return "Feature source: feature_backfill sidecar";
  if (source === "derived") return "Feature source: derived from persisted sphere-fit and diameter fields";
  if (source === "pipeline_run" || source === "pipeline_reprocess") return "Feature source: pipeline_run";
  return source ? `Feature source: ${source}` : "Feature source: unknown";
}

export function partialMissingLine(args: {
  featureKey: string;
  distribution: FeatureAnalyticsDistributionResponse | null;
  scopedObjectCount: number;
}): string | null {
  const missing = args.distribution?.stats.missing ?? 0;
  const total = args.scopedObjectCount;
  if (missing <= 0 || total <= 0) return null;
  const present = args.distribution?.stats.count ?? 0;
  if (present <= 0) return null;
  return `Feature freshness: \`${args.featureKey}\` is missing for ${missing} / ${total} objects.`;
}
