import { useEffect, useRef, useState } from "react";
import {
  api,
  type FeatureAnalyticsDistributionResponse,
  type FeatureJobRecord,
  type FeatureJobRequest,
} from "../api/client";
import {
  buildFeatureJobCliCommand,
  type FeatureFreshnessScope,
  type FeatureFreshnessState,
} from "../featureAnalyticsFreshness";

type FeatureMeta = {
  feature_key: string;
  display_name?: string;
  diagnostic_only?: boolean;
  experimental?: boolean;
};

type PendingAction = {
  mode: "reprocess" | "backfill";
  onlyMissing: boolean;
};

type Props = {
  featureKey: string;
  featureMeta: FeatureMeta | null;
  freshness: FeatureFreshnessState;
  scope: FeatureFreshnessScope | null;
  pipelineId: string | null;
  distribution: FeatureAnalyticsDistributionResponse | null;
  scopedObjectCount: number;
  actionsEnabled: { reprocess: boolean; backfill: boolean; reason?: string };
  onJobFinished: () => void;
};

function isTerminal(status: FeatureJobRecord["status"]): boolean {
  return status === "completed" || status === "failed" || status === "cancelled" || status === "partial";
}

function featureJobProcessedCount(job: FeatureJobRecord): number {
  return (job.completed_takes ?? 0) + (job.failed_takes ?? 0) + (job.skipped_takes ?? 0);
}

function featureJobProgressLine(job: FeatureJobRecord): string {
  const processed = featureJobProcessedCount(job);
  const total = job.total_takes ?? 0;
  const inFlight = job.current_take_id ? 1 : 0;
  const activeIndex = inFlight
    ? Math.min(total, processed + 1)
    : processed;
  if (job.status === "running" && inFlight) {
    return `${processed} / ${total} takes finished · working on ${activeIndex} of ${total}`;
  }
  return `${job.completed_takes ?? 0} / ${total} takes complete`;
}

export default function FeatureFreshnessPanel({
  featureKey,
  featureMeta,
  freshness,
  scope,
  pipelineId,
  distribution,
  scopedObjectCount,
  actionsEnabled,
  onJobFinished,
}: Props) {
  const [pending, setPending] = useState<PendingAction | null>(null);
  const [activeJob, setActiveJob] = useState<FeatureJobRecord | null>(null);
  const [submitError, setSubmitError] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);
  const finishedRefreshRef = useRef<string | null>(null);

  useEffect(() => {
    if (!activeJob || isTerminal(activeJob.status)) return undefined;
    let cancelled = false;
    const poll = () => {
      void api.featureJob(activeJob.id)
        .then((res) => {
          if (cancelled) return;
          setActiveJob(res.job);
        })
        .catch(() => {
          if (!cancelled) setSubmitError("Failed polling feature job status.");
        });
    };
    poll();
    const timer = window.setInterval(poll, 2000);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [activeJob?.id, activeJob?.status]);

  useEffect(() => {
    if (!activeJob || !isTerminal(activeJob.status)) return;
    if (finishedRefreshRef.current === activeJob.id) return;
    finishedRefreshRef.current = activeJob.id;
    onJobFinished();
  }, [activeJob, onJobFinished]);

  const displayName = featureMeta?.display_name || featureKey;
  const missingCount = distribution?.stats.missing ?? 0;
  const presentCount = distribution?.stats.count ?? 0;
  const partialLine = freshness === "partial_missing"
    ? `Feature freshness: \`${featureKey}\` is missing for ${missingCount} / ${scopedObjectCount} objects.`
    : null;

  const scopeLabel = scope?.mlSetName || scope?.mlSetId || scope?.datasetId || (scope?.takeIds?.length ? `${scope.takeIds.length} takes` : "current scope");
  const takeCount = scope?.takeCount ?? 0;

  async function startJob(mode: "reprocess" | "backfill", onlyMissing: boolean) {
    if (!scope) return;
    setSubmitError(null);
    const payload: FeatureJobRequest = {
      scope: scope.scope,
      dataset_id: scope.datasetId,
      ml_set_id: scope.mlSetId,
      take_ids: scope.takeIds,
      feature_key: featureKey,
      mode,
      pipeline_id: mode === "reprocess" ? pipelineId ?? undefined : undefined,
      only_missing: onlyMissing,
    };
    try {
      const response = await api.createFeatureJob(payload);
      setActiveJob(response.job);
      setPending(null);
    } catch (err: unknown) {
      setSubmitError(err instanceof Error ? err.message : "Failed to start feature job.");
    }
  }

  async function cancelJob() {
    if (!activeJob) return;
    try {
      await api.cancelFeatureJob(activeJob.id);
      const refreshed = await api.featureJob(activeJob.id);
      setActiveJob(refreshed.job);
    } catch (err: unknown) {
      setSubmitError(err instanceof Error ? err.message : "Failed to cancel feature job.");
    }
  }

  async function copyCli(mode: "reprocess" | "backfill" = "backfill") {
    if (!scope) return;
    const command = buildFeatureJobCliCommand({ mode, featureKey, scope, pipelineId });
    try {
      await navigator.clipboard.writeText(command);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1500);
    } catch {
      setCopied(false);
    }
  }

  if (freshness === "not_applicable" && !activeJob) return null;

  return (
    <div className="feature-freshness-panel">
      {freshness === "all_missing" ? (
        <div className="feature-freshness-message">
          <strong>No values for this feature in the current scope.</strong>
          <p>
            {displayName} is registered, but {scope?.mlSetId ? `ML set ${scopeLabel}` : "the current scope"} has no persisted values.
            Likely cause: these takes were processed before <code>{featureKey}</code> was emitted.
          </p>
          <div className="feature-freshness-actions">
            <button type="button" className="feature-action-primary" disabled={!actionsEnabled.reprocess || Boolean(activeJob && !isTerminal(activeJob.status))} onClick={() => setPending({ mode: "reprocess", onlyMissing: false })}>
              Reprocess ML set
            </button>
            <button type="button" className="feature-action-secondary" disabled={!actionsEnabled.backfill || Boolean(activeJob && !isTerminal(activeJob.status))} onClick={() => setPending({ mode: "backfill", onlyMissing: true })}>
              Backfill feature
            </button>
            {scope ? (
              <button type="button" className="feature-action-ghost" onClick={() => void copyCli("backfill")} disabled={!scope}>
                {copied ? "Copied CLI" : "Copy CLI command"}
              </button>
            ) : null}
          </div>
          {featureMeta?.diagnostic_only || featureMeta?.experimental ? (
            <p className="feature-freshness-warning">This feature is marked diagnostic/experimental. Use canonical reprocess for training-quality values.</p>
          ) : null}
          {!actionsEnabled.reprocess && actionsEnabled.reason ? <p className="feature-freshness-note">{actionsEnabled.reason}</p> : null}
        </div>
      ) : null}

      {freshness === "partial_missing" && partialLine ? (
        <div className="feature-freshness-partial">
          <p>{partialLine}</p>
          <div className="feature-freshness-actions">
            <button type="button" className="feature-action-primary" disabled={!actionsEnabled.reprocess || Boolean(activeJob && !isTerminal(activeJob.status))} onClick={() => setPending({ mode: "reprocess", onlyMissing: true })}>
              Reprocess missing takes
            </button>
            <button type="button" className="feature-action-secondary" disabled={!actionsEnabled.backfill || Boolean(activeJob && !isTerminal(activeJob.status))} onClick={() => setPending({ mode: "backfill", onlyMissing: true })}>
              Backfill missing values
            </button>
          </div>
        </div>
      ) : null}

      {activeJob ? (
        <div className={`feature-freshness-job-card status-${activeJob.status}`}>
          <header>
            <strong>{activeJob.mode === "reprocess" ? "Reprocessing ML set" : "Backfilling feature"}</strong>
            <span>{activeJob.status}</span>
          </header>
          <p>
            {featureJobProgressLine(activeJob)}
            {activeJob.failed_takes ? ` · Failed: ${activeJob.failed_takes}` : ""}
            {activeJob.skipped_takes ? ` · Skipped: ${activeJob.skipped_takes}` : ""}
          </p>
          {activeJob.mode === "backfill" ? (
            <p>Values written: {activeJob.values_written}{activeJob.skipped_takes ? ` · Skipped: ${activeJob.skipped_takes} already had canonical value` : ""}</p>
          ) : null}
          {activeJob.current_take_id ? <p>Current: <code>{activeJob.current_take_id}</code></p> : null}
          <p>Feature: <code>{activeJob.feature_key}</code> · Mode: {activeJob.mode}</p>
          {!isTerminal(activeJob.status) ? (
            <button type="button" className="feature-action-ghost" onClick={() => void cancelJob()}>Cancel</button>
          ) : (
            <div className="feature-freshness-actions">
              <p>
                {activeJob.mode === "reprocess" ? "Reprocess complete." : "Backfill complete."}
                {" "}
                {presentCount > 0 ? `${presentCount} objects with values` : "Refresh analytics to inspect updated counts."}
                {missingCount > 0 ? ` · ${missingCount} still missing` : ""}
              </p>
              <button type="button" className="feature-action-primary" onClick={onJobFinished}>Refresh analytics</button>
            </div>
          )}
        </div>
      ) : null}

      {submitError ? <p className="feature-freshness-error">{submitError}</p> : null}

      {pending && scope ? (
        <div className="feature-freshness-dialog-backdrop" role="presentation" onClick={() => setPending(null)}>
          <div className="feature-freshness-dialog" role="dialog" aria-modal="true" onClick={(event) => event.stopPropagation()}>
            <h3>{pending.mode === "reprocess" ? "Reprocess ML set" : "Backfill feature"}</h3>
            <dl>
              <div><dt>ML set / scope</dt><dd>{scopeLabel}</dd></div>
              <div><dt>Takes</dt><dd>{takeCount}</dd></div>
              <div><dt>Feature</dt><dd>{displayName}</dd></div>
              <div><dt>Feature key</dt><dd><code>{featureKey}</code></dd></div>
              {pending.mode === "reprocess" ? (
                <>
                  <div><dt>Pipeline</dt><dd><code>{pipelineId}</code></dd></div>
                  <div><dt>Mode</dt><dd>Canonical reprocess</dd></div>
                </>
              ) : (
                <div><dt>Mode</dt><dd>Feature backfill</dd></div>
              )}
              {pending.onlyMissing ? <div><dt>Scope filter</dt><dd>Only missing values</dd></div> : null}
            </dl>
            {pending.mode === "reprocess" ? (
              <p className="feature-freshness-dialog-copy">
                This will create new processing runs using the current pipeline. Raw takes are preserved.
                Existing outputs are not deleted unless the backend job system already has that behavior.
                Use this for training-quality feature values.
              </p>
            ) : (
              <p className="feature-freshness-dialog-copy feature-freshness-warning">
                This reads existing normalized heightmap and contour/mask artifacts and writes non-destructive feature sidecars.
                Backfilled values are useful for fast exploration but may differ from canonical reprocess values.
              </p>
            )}
            <div className="feature-freshness-actions">
              <button type="button" className="feature-action-ghost" onClick={() => setPending(null)}>Cancel</button>
              <button type="button" className="feature-action-primary" onClick={() => void startJob(pending.mode, pending.onlyMissing)}>
                {pending.mode === "reprocess" ? "Start reprocess" : "Start backfill"}
              </button>
            </div>
          </div>
        </div>
      ) : null}
    </div>
  );
}

export function featureFreshnessInspectorBlock(args: {
  freshness: FeatureFreshnessState;
  distribution: FeatureAnalyticsDistributionResponse | null;
  scopedObjectCount: number;
  featureKey: string;
  onOpenReprocess: () => void;
  onOpenBackfill: () => void;
  actionsEnabled: { reprocess: boolean; backfill: boolean };
}): { title: string; body: string; showActions: boolean } | null {
  if (args.freshness === "complete" || args.freshness === "not_applicable") return null;
  if (args.freshness === "all_missing") {
    return {
      title: "Feature freshness",
      body: "No persisted values in the current scope.",
      showActions: true,
    };
  }
  const missing = args.distribution?.stats.missing ?? 0;
  return {
    title: "Feature freshness",
    body: `${args.featureKey} missing for ${missing} / ${args.scopedObjectCount} objects.`,
    showActions: true,
  };
}
