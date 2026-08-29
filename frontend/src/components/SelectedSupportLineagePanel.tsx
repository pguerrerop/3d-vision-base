import { useEffect, useMemo, useRef, useState } from "react";
import { fileUrl, type TakeDetail } from "../api/client";
import type { StudioArtifact } from "./studioWorkspaceModel";
import type { StudioArtifactViewState } from "./studioViewState";
import StudioArtifactExplorer from "./StudioArtifactExplorer";
import InfoHint from "./InfoHint";
import { detectReferenceLineageSummaryHelpKey, resolveStudioHelpEntry } from "./studioHelp";

type Props = {
  artifacts: StudioArtifact[];
  takeId: string;
  detail: TakeDetail;
  selectedArtifactId: string | null;
  onSelectArtifact: (artifactId: string) => void;
  onFocusLineageStep?: (stepId: string, options?: { openTune?: boolean }) => void;
};

type FocusRole = "input" | "output" | "removed" | "added" | "difference" | "debug";

type LineageStep = {
  id: string;
  label: string;
  status?: string;
  enabled?: boolean;
  input_pixels?: number | null;
  output_pixels?: number | null;
  input_valid_roi_ratio?: number | null;
  output_valid_roi_ratio?: number | null;
  removed_pixels?: number | null;
  removed_fraction?: number | null;
  added_pixels?: number | null;
  added_fraction?: number | null;
  net_change_pixels?: number | null;
  net_change_fraction?: number | null;
  change_type?: string;
  primary_reason?: string;
  input_artifacts?: string[];
  output_artifacts?: string[];
  removed_artifacts?: string[];
  added_artifacts?: string[];
  parameters?: Record<string, unknown>;
};

type LineagePayload = {
  selected_surface_alias_resolves_to?: string;
  selected_surface_alias_role?: string;
  steps?: LineageStep[];
  summary?: Record<string, unknown>;
};

type FocusSelection = {
  stepId: string;
  artifactId: string | null;
  role: FocusRole;
  label: string;
};

function asRecord(value: unknown): Record<string, unknown> {
  return value && typeof value === "object" ? value as Record<string, unknown> : {};
}

function asSteps(value: unknown): LineageStep[] {
  return Array.isArray(value) ? value.filter((item): item is LineageStep => Boolean(item) && typeof item === "object") : [];
}

function formatInt(value: number | null | undefined): string {
  return Number.isFinite(value) ? Number(value).toLocaleString() : "\u2014";
}

function formatPct(value: number | null | undefined): string {
  return Number.isFinite(value) ? `${(Number(value) * 100).toFixed(1)}%` : "\u2014";
}

function signedPx(value: number | null | undefined): string {
  if (!Number.isFinite(value)) return "\u2014";
  return `${Number(value) >= 0 ? "+" : "\u2212"}${Math.abs(Number(value)).toLocaleString()} px`;
}

function signedPct(value: number | null | undefined): string {
  if (!Number.isFinite(value)) return "\u2014";
  return `${Number(value) >= 0 ? "+" : "\u2212"}${(Math.abs(Number(value)) * 100).toFixed(1)}%`;
}

function artifactImageUrl(takeId: string, artifact: StudioArtifact | null | undefined): string | null {
  return artifact?.path ? fileUrl(takeId, artifact.path) : null;
}

function preferredDifferenceArtifact(step: LineageStep): string | null {
  if ((step.removed_artifacts?.[0] ?? null) && (Number(step.removed_pixels ?? 0) > 0 || Number(step.added_pixels ?? 0) > 0)) {
    return `${step.removed_artifacts?.[0]}_overlay`;
  }
  if (step.id === "suppression_mask") return "support_removed_by_suppression_policy_overlay";
  return step.added_artifacts?.[0] ?? null;
}

function preferredDifferenceMask(step: LineageStep): { artifactId: string | null; role: FocusRole; label: string } {
  if (Number(step.removed_pixels ?? 0) > 0 && step.removed_artifacts?.[0]) {
    return { artifactId: step.removed_artifacts[0], role: "removed", label: "removed" };
  }
  if (Number(step.added_pixels ?? 0) > 0 && step.added_artifacts?.[0]) {
    return { artifactId: step.added_artifacts[0], role: "added", label: "added" };
  }
  return { artifactId: null, role: "difference", label: "difference" };
}

function changeSummary(step: LineageStep): string {
  if (step.change_type === "skipped") return "0 px / skipped";
  if (step.change_type === "alias") return "alias";
  if (step.change_type === "unknown") return "not available";
  if (!Number.isFinite(step.net_change_pixels) || !Number.isFinite(step.net_change_fraction)) return "\u2014";
  if (Number(step.net_change_pixels) === 0) return "0 px / no change";
  return `${signedPx(step.net_change_pixels)} / ${signedPct(step.net_change_fraction)}`;
}

function changeTone(step: LineageStep): "gain" | "loss" | "neutral" {
  const net = Number(step.net_change_pixels ?? 0);
  if (net > 0) return "gain";
  if (net < 0) return "loss";
  return "neutral";
}

function differenceHeadline(step: LineageStep): string {
  if (step.change_type === "expand") return `Expanded ${signedPx(step.added_pixels)}`;
  if (step.change_type === "shrink") return `Removed ${signedPx(-(step.removed_pixels ?? 0))}`;
  if (step.change_type === "mixed") return `${formatInt(step.removed_pixels)} removed / ${formatInt(step.added_pixels)} added`;
  if (step.change_type === "skipped") return "Skipped";
  if (step.change_type === "alias") return "Alias";
  return "No difference artifact";
}

function tuneKeys(stepId: string): string[] {
  if (stepId === "candidate_refinement") {
    return ["Enable refinement", "MAD k", "MAD floor mm", "Keep border support", "Minimum support pixels", "Fallback strategy"];
  }
  if (stepId === "stripe_suppression") {
    return ["Enable stripe filter", "Scope", "Threshold mode", "Min altitude mm", "Z-floor enabled", "Object kernel mm", "Max stripe height", "Warn removed fraction"];
  }
  if (stepId === "reference_model_support") {
    return ["Reference model", "Residual threshold", "Adaptive multiplier", "Minimum inlier ratio", "Expansion tolerance", "Refit after expansion"];
  }
  if (stepId === "suppression_mask") {
    return ["Suppression policy", "Suppress reference in segmentation", "Suppress stripes in segmentation"];
  }
  return [];
}

function summaryRows(summary: Record<string, unknown>): Array<{ label: string; value: string }> {
  return [
    { label: "Selected cluster", value: formatPct(Number(summary.selected_cluster_valid_roi_ratio ?? NaN)) },
    { label: "Selected support", value: formatPct(Number(summary.selected_reference_support_valid_roi_ratio ?? NaN)) },
    { label: "Model support", value: formatPct(Number(summary.reference_model_support_valid_roi_ratio ?? NaN)) },
    { label: "Suppression mask", value: formatPct(Number(summary.reference_suppression_valid_roi_ratio ?? NaN)) },
    {
      label: "Largest loss",
      value: summary.largest_support_loss_step
        ? `${String(summary.largest_support_loss_step).replace(/_/g, " ")} ${signedPct(-Number(summary.largest_support_loss_fraction ?? 0))}`
        : "\u2014",
    },
    {
      label: "Largest gain",
      value: summary.largest_support_gain_step
        ? `${String(summary.largest_support_gain_step).replace(/_/g, " ")} ${signedPct(Number(summary.largest_support_gain_fraction ?? 0))}`
        : "\u2014",
    },
  ];
}

function ThumbnailButton(props: {
  takeId: string;
  artifact: StudioArtifact | null;
  artifactId: string | null;
  label: string;
  pixels?: number | null;
  active: boolean;
  muted?: boolean;
  placeholder?: string;
  onClick: () => void;
}) {
  const src = artifactImageUrl(props.takeId, props.artifact);
  return (
    <button
      type="button"
      className={`lineage-thumb ${props.active ? "active" : ""} ${props.muted ? "muted" : ""}`}
      onClick={props.onClick}
      title={`${props.label}${props.artifactId ? ` · ${props.artifactId}` : ""}${Number.isFinite(props.pixels) ? ` · ${formatInt(props.pixels)} px` : ""}`}
    >
      <span className="lineage-thumb-label">{props.label}</span>
      {src ? <img src={src} alt={`${props.label} ${props.artifactId ?? "artifact"}`} loading="lazy" /> : <span className="lineage-thumb-placeholder">{props.placeholder ?? "Missing"}</span>}
      <span className="lineage-thumb-meta">{props.artifactId ?? "legacy / missing"}</span>
      <span className="lineage-thumb-meta">{Number.isFinite(props.pixels) ? `${formatInt(props.pixels)} px` : "\u2014"}</span>
    </button>
  );
}

export default function SelectedSupportLineagePanel({
  artifacts,
  takeId,
  detail,
  selectedArtifactId,
  onSelectArtifact,
  onFocusLineageStep,
}: Props) {
  const lineageArtifact = artifacts.find((artifact) => artifact.artifact_id === "selected_support_lineage") ?? null;
  const payload = asRecord(lineageArtifact?.metadata) as LineagePayload;
  const steps = asSteps(payload.steps);
  const artifactMap = useMemo(
    () => new Map(artifacts.map((artifact) => [artifact.artifact_id, artifact])),
    [artifacts],
  );
  const initialStepId = steps[0]?.id ?? null;
  const initialOutputId = steps[0]?.output_artifacts?.[0] ?? null;
  const [selection, setSelection] = useState<FocusSelection | null>(
    initialStepId ? { stepId: initialStepId, artifactId: initialOutputId, role: "output", label: "Output" } : null,
  );
  const [viewState, setViewState] = useState<StudioArtifactViewState>({
    viewportMode: "fit_all",
    overlayMode: "all",
    binaryMaskMode: "overlay",
    binaryMaskReferenceArtifactId: null,
    binaryMaskOpacity: 0.62,
    binaryMaskReferenceOpacity: 1,
  });
  const previewSectionRef = useRef<HTMLDivElement | null>(null);

  const activeStep = steps.find((step) => step.id === selection?.stepId) ?? steps[0] ?? null;

  const previewArtifacts = useMemo(() => {
    const ids = [
      selection?.artifactId ?? null,
      activeStep?.input_artifacts?.[0] ?? null,
      activeStep?.output_artifacts?.[0] ?? null,
      activeStep?.removed_artifacts?.[0] ?? null,
      activeStep?.added_artifacts?.[0] ?? null,
      activeStep ? preferredDifferenceArtifact(activeStep) : null,
      "selected_support_lineage",
    ].filter((value): value is string => Boolean(value));
    return Array.from(new Set(ids)).map((id) => artifactMap.get(id)).filter((artifact): artifact is StudioArtifact => Boolean(artifact));
  }, [activeStep, artifactMap, selection?.artifactId]);

  useEffect(() => {
    if (!selection?.artifactId || !steps.length) return;
    previewSectionRef.current?.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }, [selection?.artifactId, steps.length]);

  useEffect(() => {
    if (!selectedArtifactId || selection?.artifactId || !steps.length) return;
    setSelection({ stepId: steps[0]?.id ?? "", artifactId: selectedArtifactId, role: "output", label: "Output" });
  }, [selectedArtifactId, selection?.artifactId, steps]);

  useEffect(() => {
    setViewState((current) => ({
      ...current,
      binaryMaskReferenceArtifactId: null,
    }));
  }, [selection?.artifactId]);

  if (!lineageArtifact) {
    return (
      <div className="empty-state">
        <strong>Selected support lineage was not emitted for this run.</strong>
        <small>Rerun the take to generate lineage diagnostics.</small>
      </div>
    );
  }

  if (!steps.length) {
    return (
      <div className="empty-state">
        <strong>No selected-support lineage steps are available.</strong>
        <small>Numeric diagnostics may exist, but this run did not emit per-step lineage entries.</small>
      </div>
    );
  }

  const selectStepArtifact = (
    step: LineageStep,
    role: FocusRole,
    artifactId: string | null,
    label: string,
    openTune = true,
  ) => {
    setSelection({ stepId: step.id, artifactId, role, label });
    if (artifactId) onSelectArtifact(artifactId);
    onFocusLineageStep?.(step.id, { openTune });
  };

  const focusedPixels =
    selection?.role === "input" ? activeStep?.input_pixels
      : selection?.role === "output" ? activeStep?.output_pixels
        : selection?.role === "removed" ? activeStep?.removed_pixels
          : selection?.role === "added" ? activeStep?.added_pixels
            : selection?.role === "difference" ? activeStep?.net_change_pixels
              : null;

  return (
    <div className="selected-support-lineage-panel">
      <div className="artifact-reference">
        <span>Selected support lineage</span>
        <small>
          Selected surface alias: {payload.selected_surface_alias_resolves_to ?? "Not available"}
          {payload.selected_surface_alias_role ? ` (${payload.selected_surface_alias_role})` : ""}
        </small>
        <small>Focused artifact: {selection?.artifactId ?? "none selected"}</small>
      </div>

      <div className="selected-support-lineage-summary">
        {summaryRows(asRecord(payload.summary)).map((row) => (
          <div key={row.label} className="selected-support-lineage-summary-card">
            <span className="info-inline-label">
              <span>{row.label}</span>
              <InfoHint
                content={resolveStudioHelpEntry(detectReferenceLineageSummaryHelpKey(row.label))}
                label="Open summary help"
              />
            </span>
            <strong>{row.value}</strong>
          </div>
        ))}
      </div>

      <section className="selected-support-lineage-filmstrip">
        <div className="artifact-reference">
          <span>Mask evolution</span>
          <small>Click a card to focus the step in the ledger and preview.</small>
        </div>
        <div className="selected-support-lineage-filmstrip-row">
          {steps.map((step, index) => {
            const outputArtifactId = step.output_artifacts?.[0] ?? null;
            const outputArtifact = outputArtifactId ? artifactMap.get(outputArtifactId) ?? null : null;
            return (
              <button
                key={step.id}
                type="button"
                className={`lineage-film-card ${selection?.stepId === step.id ? "active" : ""}`}
                onClick={() => selectStepArtifact(step, "output", outputArtifactId, "Output")}
              >
                <span className="lineage-film-card-title">{step.label}</span>
                {artifactImageUrl(takeId, outputArtifact) ? (
                  <img src={artifactImageUrl(takeId, outputArtifact) ?? ""} alt={step.label} loading="lazy" />
                ) : (
                  <span className="lineage-film-card-placeholder">No output artifact emitted for this step.</span>
                )}
                <small>{formatInt(step.output_pixels)} px</small>
                <small>{formatPct(step.output_valid_roi_ratio)} valid ROI</small>
                <small className={`tone-${changeTone(step)}`}>{changeSummary(step)}</small>
                <small>{step.status ?? (step.enabled ? "ok" : "skipped")}</small>
                <small>{step.primary_reason ?? "\u2014"}</small>
                {index < steps.length - 1 ? <span className="lineage-film-arrow" aria-hidden="true">\u2192</span> : null}
              </button>
            );
          })}
        </div>
      </section>

      <div className="selected-support-lineage-layout">
        <div className="selected-support-lineage-table-pane">
          <table className="compact-candidate-table selected-support-lineage-ledger">
            <thead>
              <tr>
                <th>Step</th>
                <th>Input</th>
                <th>Output</th>
                <th>Removed / Added</th>
                <th>Change</th>
                <th>Status</th>
                <th>Reason</th>
                <th>Tune</th>
                <th>Debug</th>
              </tr>
            </thead>
            <tbody>
              {steps.map((step) => {
                const inputArtifactId = step.input_artifacts?.[0] ?? null;
                const outputArtifactId = step.output_artifacts?.[0] ?? null;
                const diffMask = preferredDifferenceMask(step);
                const diffArtifact = diffMask.artifactId ? artifactMap.get(diffMask.artifactId) ?? null : null;
                return (
                  <tr
                    key={step.id}
                    className={selection?.stepId === step.id ? "selected" : ""}
                    onClick={() => selectStepArtifact(step, "output", outputArtifactId, "Output")}
                  >
                    <td>
                      <strong>{step.label}</strong>
                      <small>{differenceHeadline(step)}</small>
                    </td>
                    <td>
                      <ThumbnailButton
                        takeId={takeId}
                        artifact={inputArtifactId ? artifactMap.get(inputArtifactId) ?? null : null}
                        artifactId={inputArtifactId}
                        label="input"
                        pixels={step.input_pixels}
                        active={selection?.stepId === step.id && selection.role === "input"}
                        muted={!inputArtifactId}
                        placeholder="Missing input"
                        onClick={() => selectStepArtifact(step, "input", inputArtifactId, "Input")}
                      />
                    </td>
                    <td>
                      <ThumbnailButton
                        takeId={takeId}
                        artifact={outputArtifactId ? artifactMap.get(outputArtifactId) ?? null : null}
                        artifactId={outputArtifactId}
                        label="output"
                        pixels={step.output_pixels}
                        active={selection?.stepId === step.id && selection.role === "output"}
                        muted={!outputArtifactId}
                        placeholder="Missing output"
                        onClick={() => selectStepArtifact(step, "output", outputArtifactId, "Output")}
                      />
                    </td>
                    <td>
                      <ThumbnailButton
                        takeId={takeId}
                        artifact={diffArtifact}
                        artifactId={diffMask.artifactId}
                        label={diffMask.label}
                        pixels={diffMask.role === "removed" ? step.removed_pixels : step.added_pixels}
                        active={selection?.stepId === step.id && selection.role === diffMask.role}
                        muted={!diffMask.artifactId}
                        placeholder="No diff artifact"
                        onClick={() => selectStepArtifact(step, diffMask.role, diffMask.artifactId, diffMask.label)}
                      />
                    </td>
                    <td className={`tone-${changeTone(step)}`}>
                      <strong>{changeSummary(step)}</strong>
                      <small>{step.change_type ?? "unknown"}</small>
                    </td>
                    <td>{step.status ?? (step.enabled ? "ok" : "skipped")}</td>
                    <td>{step.primary_reason ?? "\u2014"}</td>
                    <td>
                      <div className="lineage-tune-cell">
                        <button type="button" onClick={(event) => { event.stopPropagation(); onFocusLineageStep?.(step.id, { openTune: true }); }}>
                          Tune
                        </button>
                        <small>{tuneKeys(step.id).slice(0, 3).join(" • ") || "No step-specific controls"}</small>
                      </div>
                    </td>
                    <td>
                      <div className="overlay-controls">
                        <button type="button" onClick={(event) => { event.stopPropagation(); selectStepArtifact(step, "input", inputArtifactId, "Input"); }}>View input</button>
                        <button type="button" onClick={(event) => { event.stopPropagation(); selectStepArtifact(step, "output", outputArtifactId, "Output"); }}>View output</button>
                        <button type="button" onClick={(event) => { event.stopPropagation(); selectStepArtifact(step, "difference", preferredDifferenceArtifact(step), "Difference"); }}>View removed</button>
                        <button type="button" onClick={(event) => { event.stopPropagation(); selectStepArtifact(step, "debug", "selected_support_lineage", "Debug JSON", false); }}>Debug JSON</button>
                      </div>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>

          {activeStep ? (
            <div className="artifact-reference selected-support-lineage-step-note">
              <span>{activeStep.label}</span>
              <small>{activeStep.primary_reason ?? "No reason recorded."}</small>
              <small>Affects tuning: {tuneKeys(activeStep.id).join(" • ") || "No step-specific controls recorded."}</small>
            </div>
          ) : null}
        </div>

        <div className="selected-support-lineage-preview-pane" ref={previewSectionRef}>
          <div className="artifact-reference">
            <span>Selected lineage artifact</span>
            <small>Step: {activeStep?.label ?? "Not available"}</small>
            <small>Role: {selection?.label ?? "Not available"}</small>
            <small>Artifact: {selection?.artifactId ?? "Not available"}</small>
            <small>Reason: {activeStep?.primary_reason ?? "Not available"}</small>
            <small>Pixels: {formatInt(focusedPixels)}</small>
            <small>Change: {activeStep ? changeSummary(activeStep) : "\u2014"}</small>
            <small>Legend: Red = removed from previous step · Blue = added by this step · Green = retained</small>
          </div>
          {previewArtifacts.length ? (
            <StudioArtifactExplorer
              artifacts={previewArtifacts}
              takeId={takeId}
              selectedArtifactId={selection?.artifactId ?? null}
              onSelect={(artifactId) => {
                setSelection((current) => current ? { ...current, artifactId } : current);
                onSelectArtifact(artifactId);
              }}
              detailJson={detail.result}
              takeDetail={detail}
              currentStageArtifacts={previewArtifacts}
              selectedObjectId={null}
              hoveredObjectId={null}
              hoveredArtifactId={null}
              onSelectObject={() => {}}
              onHoverObject={() => {}}
              onHoverArtifact={() => {}}
              onOverlayDebug={() => {}}
              roi={null}
              roiPolygon={null}
              roiEnabled={false}
              roiEditModeActive={false}
              roiEditStatus="idle"
              debugMode="engineering"
              onHoverSample={() => {}}
              viewState={viewState}
              onViewportModeChange={(mode) => setViewState((current) => ({ ...current, viewportMode: mode }))}
              onOverlayModeChange={(mode) => setViewState((current) => ({ ...current, overlayMode: mode }))}
              onBinaryMaskModeChange={(mode) => setViewState((current) => ({ ...current, binaryMaskMode: mode }))}
              onBinaryMaskReferenceArtifactIdChange={(artifactId) => setViewState((current) => ({ ...current, binaryMaskReferenceArtifactId: artifactId }))}
              onBinaryMaskOpacityChange={(value) => setViewState((current) => ({ ...current, binaryMaskOpacity: value }))}
              onBinaryMaskReferenceOpacityChange={(value) => setViewState((current) => ({ ...current, binaryMaskReferenceOpacity: value }))}
              onResetViewState={() => setViewState({
                viewportMode: "fit_all",
                overlayMode: "all",
                binaryMaskMode: "overlay",
                binaryMaskReferenceArtifactId: null,
                binaryMaskOpacity: 0.62,
                binaryMaskReferenceOpacity: 1,
              })}
            />
          ) : (
            <div className="empty-state">
              <strong>No preview artifact available for this action.</strong>
              <small>This row is still kept so numeric diagnostics remain visible for older or partial runs.</small>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
