import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api, fileUrl, type PipelineInfo, type TakeDetail } from "../api/client";
import BinaryMaskRenderer from "../components/BinaryMaskRenderer";
import {
  defaultBinaryMaskViewMode,
  resolveBinaryMaskReferenceCandidates,
  shouldUseBinaryMaskRenderer,
} from "../components/binaryMaskArtifacts";
import { artifactRelevanceLabel, artifactRelevanceTone } from "../components/referenceArtifactRelevance";
import IdMaskRenderer from "../components/IdMaskRenderer";
import { isIdMaskArtifact } from "../components/idMaskArtifacts";
import { sourceArtifactsForTake } from "../components/stage_sources";
import { stageSemanticDefinition } from "../components/stageSemantics";
import { preferredViewportModeForArtifact, type ViewportMode } from "../components/studioViewState";
import { artifactList, artifactsForStage, type StudioArtifact } from "../components/studioWorkspaceModel";
import BlobSetViewer from "../components/BlobSetViewer";
import { normalizeBlobDetectionArtifacts, resolveBlobSetSourceImage } from "../studio/blobDetectionModel";
import { useLowGradientComponents } from "../studio/lowGradientComponentModel";
import { buildStudioDeepLink } from "../studioDeepLink";
import {
  buildPipelineWalkthrough,
  type WalkthroughArtifact,
  type WalkthroughArtifactProvenance,
  type WalkthroughParameter,
  type WalkthroughParameterGroup,
  type WalkthroughStage,
  type WalkthroughSubstage,
} from "../components/pipelineWalkthroughModel";

type LightboxState = { src: string; label: string; artifact: WalkthroughArtifact; tuneHref: string } | null;
type WalkthroughImageMode = "single" | "overlay" | "side_by_side";

function queryParams(): URLSearchParams {
  return new URLSearchParams(window.location.search);
}

function resolveArtifactUrl(takeId: string, artifact: StudioArtifact | null): string | null {
  if (!artifact) return null;
  const external = String((artifact.metadata as Record<string, unknown> | undefined)?.external_source_url ?? "").trim();
  if (/^https?:\/\//i.test(external)) return external;
  if (artifact.path) return fileUrl(takeId, artifact.path);
  return null;
}

function viewportClass(fitMode: ViewportMode): string {
  if (fitMode === "fit_width") return "viewport-fit-width";
  if (fitMode === "fit_height") return "viewport-fit-height";
  if (fitMode === "natural") return "viewport-natural";
  return "viewport-fit-all";
}

function RelevanceBadge({ relevance, feedsInto }: { relevance: WalkthroughArtifact["relevance"]; feedsInto?: WalkthroughArtifact["feedsInto"] }) {
  if (relevance === "active") return null;
  const tone = artifactRelevanceTone(relevance);
  const title = relevance === "debug" && feedsInto && feedsInto.length > 0
    ? `Used while producing: ${feedsInto.map((entry) => entry.label).join(", ")}`
    : undefined;
  return <span className={`artifact-relevance-badge tone-${tone}`} title={title}>{artifactRelevanceLabel(relevance)}</span>;
}

function IoRoleTag({ ioRole, feedsInto }: { ioRole: WalkthroughArtifact["ioRole"]; feedsInto?: WalkthroughArtifact["feedsInto"] }) {
  if (ioRole !== "input" && ioRole !== "output") return null;
  // OUT covers two cases with one badge: a declared role="final" hand-off, or a diagnostic verified
  // to transitively reach a real final/input artifact outside this substage. feedsInto is only ever
  // populated for the second case, so use it to say exactly where when we have it.
  const title = ioRole === "input"
    ? "Consumed from an earlier substage"
    : feedsInto && feedsInto.length > 0
      ? `Reaches: ${feedsInto.map((entry) => entry.label).join(", ")}`
      : "Used by a later step";
  return <span className={`artifact-io-role-tag role-${ioRole}`} title={title}>{ioRole === "input" ? "in" : "out"}</span>;
}

function formatArtifactParameter(param: WalkthroughArtifactProvenance["parameters"][number]): string {
  return param.formattedValue ?? "Not set";
}

type ProvenanceTab = "method" | "inputs" | "parameters";

function ArtifactProvenance({ provenance }: { provenance?: WalkthroughArtifactProvenance }) {
  const [activeTab, setActiveTab] = useState<ProvenanceTab>("method");

  useEffect(() => {
    setActiveTab("method");
  }, [provenance?.calculatedBy, provenance?.calculation]);

  if (!provenance) return null;
  return (
    <aside className="walkthrough-artifact-provenance" aria-label="Artifact calculation details">
      <div className="walkthrough-artifact-provenance-tabs" role="tablist" aria-label="Artifact explanation sections">
        <button type="button" role="tab" aria-selected={activeTab === "method"} className={activeTab === "method" ? "active" : ""} onClick={() => setActiveTab("method")}>Method</button>
        <button type="button" role="tab" aria-selected={activeTab === "inputs"} className={activeTab === "inputs" ? "active" : ""} onClick={() => setActiveTab("inputs")}>Inputs</button>
        <button type="button" role="tab" aria-selected={activeTab === "parameters"} className={activeTab === "parameters" ? "active" : ""} onClick={() => setActiveTab("parameters")}>Parameters</button>
      </div>

      {activeTab === "method" ? (
        <div className="walkthrough-artifact-provenance-panel" role="tabpanel">
          <div className="walkthrough-artifact-summary-card">
            <span>What this output is</span>
            <p>{provenance.calculation}</p>
          </div>
          <div>
            <span>Producing method</span>
            <strong>{provenance.calculatedBy}</strong>
            <p>{provenance.method}</p>
          </div>
          <div>
            <span>How it is produced</span>
            <ol className="walkthrough-artifact-step-list">
              {provenance.methodSteps.map((step, index) => (
                <li key={`${index}-${step}`}>{step}</li>
              ))}
            </ol>
          </div>
        </div>
      ) : null}

      {activeTab === "inputs" ? (
        <div className="walkthrough-artifact-provenance-panel" role="tabpanel">
          <div>
            <span>Upstream inputs</span>
            {provenance.inputs.length > 0 ? (
              <div className="walkthrough-artifact-input-list">
                {provenance.inputs.map((input) => (
                  <article key={input.id} className="walkthrough-artifact-input-card">
                    <div className="walkthrough-artifact-input-heading">
                      <strong>{input.label}</strong>
                      <small>{input.required ? "Required" : "Optional"}</small>
                    </div>
                    <p>{input.description ?? "Upstream artifact consumed by this method."}</p>
                    <p className="walkthrough-artifact-input-usage">{input.usage ?? `Used by ${provenance.calculatedBy.toLowerCase()} while producing this artifact.`}</p>
                  </article>
                ))}
              </div>
            ) : <p>No upstream artifact inputs are declared for this step.</p>}
          </div>
        </div>
      ) : null}

      {activeTab === "parameters" ? (
        <div className="walkthrough-artifact-provenance-panel" role="tabpanel">
          <div>
            <span>Parameters affecting this artifact</span>
            {provenance.parameters.length > 0 ? (
              <ul className="walkthrough-artifact-parameter-list">
                {provenance.parameters.map((param) => (
                  <li key={param.id} className={param.relevant ? undefined : "deemphasized"} title={param.relevanceReason}>
                    <div>
                      <strong>{param.label}</strong>
                      {param.tuningHint ? <p>{param.tuningHint}</p> : null}
                    </div>
                    <small>{formatArtifactParameter(param)}</small>
                  </li>
                ))}
              </ul>
            ) : <p>This calculation has no tunable parameters.</p>}
          </div>
        </div>
      ) : null}
    </aside>
  );
}

function Lightbox({ state, detail, takeId, onClose }: { state: LightboxState; detail: TakeDetail | null; takeId: string; onClose: () => void }) {
  const [imageMode, setImageMode] = useState<WalkthroughImageMode>("single");
  const [fitMode, setFitMode] = useState<ViewportMode>("fit_all");
  const [overlayOpacity, setOverlayOpacity] = useState(0.52);
  const [referenceOpacity, setReferenceOpacity] = useState(1);
  const [selectedReferenceId, setSelectedReferenceId] = useState<string | null>(null);
  const [binaryMaskMode, setBinaryMaskMode] = useState<"mask" | "overlay" | "boundary" | "side_by_side">("mask");
  const [selectedBlobId, setSelectedBlobId] = useState<number | null>(null);
  const [hoveredBlobId, setHoveredBlobId] = useState<number | null>(null);
  const initializedArtifactIdRef = useRef<string | null>(null);

  useEffect(() => {
    if (!state) return;
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [state, onClose]);

  const selectedArtifact = state?.artifact.artifact ?? null;
  const allArtifacts = useMemo(
    () => artifactList(detail),
    [detail],
  );
  const stageArtifacts = useMemo(
    () => (selectedArtifact?.stage_id ? artifactsForStage(detail, selectedArtifact.stage_id) : []),
    [detail, selectedArtifact?.stage_id],
  );
  const sourceArtifacts = useMemo(
    () => sourceArtifactsForTake(detail, selectedArtifact?.stage_id ?? "input"),
    [detail, selectedArtifact?.stage_id],
  );
  const referenceCandidates = useMemo(
    () => selectedArtifact ? resolveBinaryMaskReferenceCandidates({
      maskArtifact: selectedArtifact,
      selectedStageId: selectedArtifact.stage_id,
      selectedTake: detail,
      currentStageArtifacts: stageArtifacts,
      allRunArtifacts: allArtifacts,
      sourceArtifacts,
      resolvedArtifacts: [...stageArtifacts, ...sourceArtifacts],
    }) : [],
    [allArtifacts, detail, selectedArtifact, sourceArtifacts, stageArtifacts],
  );
  const selectedReference = useMemo(
    () => referenceCandidates.find((candidate) => candidate.artifact.artifact_id === selectedReferenceId) ?? referenceCandidates[0] ?? null,
    [referenceCandidates, selectedReferenceId],
  );
  const selectedReferenceUrl = useMemo(
    () => selectedReference?.sourceUrl ?? resolveArtifactUrl(takeId, selectedReference?.artifact ?? null),
    [selectedReference?.artifact, selectedReference?.sourceUrl, takeId],
  );
  const previewUrl = useMemo(
    () => resolveArtifactUrl(takeId, selectedArtifact) ?? state?.src ?? null,
    [selectedArtifact, state?.src, takeId],
  );
  const useBinaryMaskView = Boolean(selectedArtifact && shouldUseBinaryMaskRenderer(selectedArtifact, allArtifacts));
  const useIdMaskView = Boolean(selectedArtifact && isIdMaskArtifact(selectedArtifact));
  const blobSetData = useMemo(
    () => (selectedArtifact && stageSemanticDefinition(selectedArtifact.stage_id).category === "geometry"
      ? normalizeBlobDetectionArtifacts(stageArtifacts)
      : null),
    [selectedArtifact, stageArtifacts],
  );
  const useBlobSetView = Boolean(selectedArtifact?.kind === "image" && blobSetData && blobSetData.candidates.length > 0);
  const blobSetSourceUrl = useMemo(
    () => (useBlobSetView ? resolveArtifactUrl(takeId, resolveBlobSetSourceImage(stageArtifacts)) ?? previewUrl : null),
    [useBlobSetView, stageArtifacts, takeId, previewUrl],
  );
  const surfaceCandidatesArtifact = useMemo(
    () => stageArtifacts.find((item) => item.artifact_id === "reference_surface_candidates") ?? null,
    [stageArtifacts],
  );
  const lowGradientComponents = useLowGradientComponents(surfaceCandidatesArtifact?.path ? fileUrl(takeId, surfaceCandidatesArtifact.path) : null);
  const useLowGradientComponentView = Boolean(
    selectedArtifact?.kind === "image"
    && selectedArtifact.stage_id
    && stageSemanticDefinition(selectedArtifact.stage_id).category === "plane_qa"
    && surfaceCandidatesArtifact
    && !lowGradientComponents.loading
    && !lowGradientComponents.error
    && [...lowGradientComponents.components, ...lowGradientComponents.rejected].some((row) => row.bbox && row.centroid),
  );
  const lowGradientComponentsSourceUrl = useMemo(
    () => (useLowGradientComponentView
      ? resolveArtifactUrl(
        takeId,
        stageArtifacts.find((item) => item.artifact_id === "raw_heightmap_preview" && item.path)
          ?? stageArtifacts.find((item) => item.artifact_id === "low_gradient_components_overlay" && item.path)
          ?? null,
      ) ?? previewUrl
      : null),
    [useLowGradientComponentView, stageArtifacts, takeId, previewUrl],
  );
  const lowGradientComponentsMaskSourceUrl = useMemo(
    () => (useLowGradientComponentView
      ? resolveArtifactUrl(takeId, stageArtifacts.find((item) => item.artifact_id === "low_gradient_blob_id_mask" && item.path) ?? null)
      : null),
    [useLowGradientComponentView, stageArtifacts, takeId],
  );

  useEffect(() => {
    if (!state) return;
    setImageMode("single");
    setBinaryMaskMode("mask");
    setFitMode(preferredViewportModeForArtifact(state.artifact.artifact ?? null));
    setOverlayOpacity(0.52);
    setReferenceOpacity(1);
    setSelectedReferenceId(null);
    setSelectedBlobId(null);
    setHoveredBlobId(null);
  }, [state?.artifact.artifactId]);

  useEffect(() => {
    const fallbackId = referenceCandidates[0]?.artifact.artifact_id ?? null;
    if (selectedReferenceId && referenceCandidates.some((candidate) => candidate.artifact.artifact_id === selectedReferenceId)) return;
    if (selectedReferenceId === fallbackId) return;
    setSelectedReferenceId(fallbackId);
  }, [referenceCandidates, selectedReferenceId]);

  useEffect(() => {
    const artifactId = state?.artifact.artifactId ?? null;
    if (!artifactId || initializedArtifactIdRef.current === artifactId) return;
    if (useBinaryMaskView && selectedArtifact) {
      setBinaryMaskMode(defaultBinaryMaskViewMode(selectedArtifact, allArtifacts));
      initializedArtifactIdRef.current = artifactId;
      return;
    }
    if (!useBinaryMaskView && selectedReferenceUrl) {
      setImageMode("overlay");
      initializedArtifactIdRef.current = artifactId;
      return;
    }
    if (!useBinaryMaskView && !referenceCandidates.length) {
      initializedArtifactIdRef.current = artifactId;
    }
  }, [allArtifacts, referenceCandidates.length, selectedArtifact, selectedReferenceUrl, state?.artifact.artifactId, useBinaryMaskView]);

  if (!state) return null;
  return (
    <div className="walkthrough-lightbox" onClick={onClose} role="dialog" aria-modal="true" aria-label={state.label}>
      <div className="walkthrough-lightbox-content" onClick={(event) => event.stopPropagation()}>
        <figure>
          <div className="walkthrough-lightbox-visual-shell">
            {selectedArtifact?.kind === "image" ? (
              <>
                {useLowGradientComponentView ? (
                  <BlobSetViewer
                    src={lowGradientComponentsSourceUrl}
                    maskSourceSrc={lowGradientComponentsMaskSourceUrl}
                    candidates={lowGradientComponents.components}
                    rejected={lowGradientComponents.rejected}
                    selectedCandidateId={selectedBlobId}
                    hoveredCandidateId={hoveredBlobId}
                    onSelectCandidate={setSelectedBlobId}
                    onHoverCandidate={setHoveredBlobId}
                  />
                ) : useBlobSetView && blobSetData ? (
                  <BlobSetViewer
                    src={blobSetSourceUrl}
                    candidates={blobSetData.candidates}
                    rejected={blobSetData.rejected}
                    selectedCandidateId={selectedBlobId}
                    hoveredCandidateId={hoveredBlobId}
                    onSelectCandidate={setSelectedBlobId}
                    onHoverCandidate={setHoveredBlobId}
                  />
                ) : useIdMaskView ? (
                  <div className="overlay-frame">
                    <IdMaskRenderer src={previewUrl ?? ""} alt={state.label} />
                  </div>
                ) : useBinaryMaskView ? (
                  <BinaryMaskRenderer
                    artifact={selectedArtifact}
                    artifacts={allArtifacts}
                    cacheKey={selectedArtifact.created_at ?? selectedArtifact.path ?? selectedArtifact.artifact_id}
                    currentStageArtifacts={stageArtifacts}
                    fitMode={fitMode}
                    mode={binaryMaskMode}
                    onFitModeChange={setFitMode}
                    onModeChange={setBinaryMaskMode}
                    onOverlayOpacityChange={setOverlayOpacity}
                    onReferenceOpacityChange={setReferenceOpacity}
                    onResetView={() => {
                      setFitMode(preferredViewportModeForArtifact(selectedArtifact));
                      setBinaryMaskMode("mask");
                      setOverlayOpacity(0.52);
                      setReferenceOpacity(1);
                    }}
                    onSelectedSourceArtifactIdChange={setSelectedReferenceId}
                    overlayOpacity={overlayOpacity}
                    openUrl={previewUrl}
                    referenceOpacity={referenceOpacity}
                    selectedSourceArtifactId={selectedReferenceId}
                    selectedTake={detail}
                    takeId={takeId}
                  />
                ) : (
                  <>
                    <div className="walkthrough-lightbox-visual-toolbar">
                      <div className="walkthrough-lightbox-visual-group">
                        <button className={imageMode === "single" ? "active" : ""} onClick={() => setImageMode("single")} type="button">Image</button>
                        <button className={imageMode === "overlay" ? "active" : ""} disabled={!selectedReferenceUrl} onClick={() => setImageMode("overlay")} type="button">Overlay</button>
                        <button className={imageMode === "side_by_side" ? "active" : ""} disabled={!selectedReferenceUrl} onClick={() => setImageMode("side_by_side")} type="button">Side-by-side</button>
                      </div>
                      <div className="walkthrough-lightbox-visual-group">
                        <button className={fitMode === "fit_all" ? "active" : ""} onClick={() => setFitMode("fit_all")} type="button">Fit all</button>
                        <button className={fitMode === "fit_width" ? "active" : ""} onClick={() => setFitMode("fit_width")} type="button">Fit width</button>
                        <button className={fitMode === "fit_height" ? "active" : ""} onClick={() => setFitMode("fit_height")} type="button">Fit height</button>
                        <button className={fitMode === "natural" ? "active" : ""} onClick={() => setFitMode("natural")} type="button">100%</button>
                      </div>
                    </div>
                    {referenceCandidates.length > 0 ? (
                      <div className="walkthrough-lightbox-visual-meta">
                        <label>
                          <span>Reference</span>
                          <select value={selectedReferenceId ?? ""} onChange={(event) => setSelectedReferenceId(event.target.value || null)}>
                            {referenceCandidates.map((candidate) => (
                              <option key={candidate.artifact.artifact_id} value={candidate.artifact.artifact_id}>{candidate.label}</option>
                            ))}
                          </select>
                        </label>
                        {imageMode === "overlay" && selectedReferenceUrl ? (
                          <>
                            <label>
                              <span>Overlay opacity</span>
                              <input max="100" min="0" onChange={(event) => setOverlayOpacity(Number(event.target.value) / 100)} type="range" value={Math.round(overlayOpacity * 100)} />
                            </label>
                            <label>
                              <span>Reference opacity</span>
                              <input max="100" min="0" onChange={(event) => setReferenceOpacity(Number(event.target.value) / 100)} type="range" value={Math.round(referenceOpacity * 100)} />
                            </label>
                          </>
                        ) : null}
                      </div>
                    ) : null}
                    <div className={`walkthrough-lightbox-visual ${viewportClass(fitMode)}`}>
                      {imageMode === "side_by_side" && selectedReferenceUrl && previewUrl ? (
                        <div className="walkthrough-lightbox-side-by-side">
                          <div className="walkthrough-lightbox-side-pane">
                            <strong>{selectedReference?.label ?? "Reference"}</strong>
                            <div className="overlay-frame">
                              <img alt={selectedReference?.label ?? "Reference"} src={selectedReferenceUrl} />
                            </div>
                          </div>
                          <div className="walkthrough-lightbox-side-pane">
                            <strong>{state.label}</strong>
                            <div className="overlay-frame">
                              <img alt={state.label} src={previewUrl} />
                            </div>
                          </div>
                        </div>
                      ) : (
                        <div className="overlay-frame">
                          {imageMode === "overlay" && selectedReferenceUrl && previewUrl ? (
                            <>
                              <img alt={selectedReference?.label ?? "Reference"} src={selectedReferenceUrl} style={{ opacity: referenceOpacity }} />
                              <img alt={state.label} className="walkthrough-lightbox-overlay-image" src={previewUrl} style={{ opacity: overlayOpacity }} />
                            </>
                          ) : previewUrl ? (
                            <img alt={state.label} src={previewUrl} />
                          ) : null}
                        </div>
                      )}
                    </div>
                  </>
                )}
              </>
            ) : previewUrl ? <img alt={state.label} src={previewUrl} /> : null}
          </div>
          <figcaption>{state.label}</figcaption>
        </figure>
        <div className="walkthrough-lightbox-details">
          <div className="walkthrough-lightbox-heading">
            <div>
              <h2>{state.label}</h2>
              <IoRoleTag ioRole={state.artifact.ioRole} feedsInto={state.artifact.feedsInto} />
              <RelevanceBadge relevance={state.artifact.relevance} feedsInto={state.artifact.feedsInto} />
            </div>
            <button type="button" onClick={onClose} aria-label="Close">✕</button>
          </div>
          <ArtifactProvenance provenance={state.artifact.provenance} />
          <a className="walkthrough-lightbox-tune-link" href={state.tuneHref} target="_blank" rel="noreferrer">Tune these parameters ↗</a>
        </div>
      </div>
    </div>
  );
}

function ArtifactChip({ takeId, entry, tuneHref, onOpen }: { takeId: string; entry: WalkthroughArtifact; tuneHref: string; onOpen: (state: LightboxState) => void }) {
  // Dimming signals "not relevant to this run" (a different strategy branch, or genuinely
  // absent), not "diagnostic role" -- used artifacts are real, computed data and should read
  // at full contrast; the USED badge alone is enough to convey their role.
  const isDeemphasized = entry.relevance === "inactive_strategy" || entry.relevance === "legacy";
  const image = entry.artifact?.kind === "image" && entry.artifact.path ? fileUrl(takeId, entry.artifact.path) : null;
  const isIdMask = Boolean(entry.artifact && isIdMaskArtifact(entry.artifact));
  return (
    <div className={`walkthrough-artifact-chip${isDeemphasized ? " deemphasized" : ""}`} title={entry.reason ?? undefined}>
      {image ? (
        <button type="button" className="walkthrough-artifact-chip-image-button" onClick={() => onOpen({ src: image, label: entry.label, artifact: entry, tuneHref })}>
          {isIdMask ? <IdMaskRenderer src={image} alt={entry.label} /> : <img src={image} alt={entry.label} loading="lazy" />}
        </button>
      ) : null}
      <div className="walkthrough-artifact-chip-label">
        <span>{entry.label}</span>
        <IoRoleTag ioRole={entry.ioRole} feedsInto={entry.feedsInto} />
        <RelevanceBadge relevance={entry.relevance} feedsInto={entry.feedsInto} />
      </div>
    </div>
  );
}

function ParameterRow({ param }: { param: WalkthroughParameter }) {
  const differsFromDefault = param.value !== undefined && param.value !== param.default;
  return (
    <tr className={param.relevant ? undefined : "deemphasized"}>
      <td>
        {param.label}
        {!param.relevant ? <span className="walkthrough-na-tag" title={param.relevanceReason}>N/A for current strategy</span> : null}
      </td>
      <td className="walkthrough-value-cell">
        {param.formattedValue ?? <span className="walkthrough-empty-value">—</span>}
        {differsFromDefault && param.relevant ? <small> (default: {String(param.default)})</small> : null}
      </td>
      <td className="walkthrough-hint-cell">{param.relevant ? param.tuningHint ?? "" : param.relevanceReason ?? ""}</td>
    </tr>
  );
}

function ParameterTable({ params }: { params: WalkthroughParameter[] }) {
  if (params.length === 0) return null;
  return (
    <table className="walkthrough-parameter-table">
      <thead>
        <tr>
          <th>Parameter</th>
          <th>Current value</th>
          <th>Tuning notes</th>
        </tr>
      </thead>
      <tbody>
        {params.map((param) => <ParameterRow key={param.id} param={param} />)}
      </tbody>
    </table>
  );
}

function ParameterGroupBlock({ group }: { group: WalkthroughParameterGroup }) {
  return (
    <div className="walkthrough-parameter-group">
      <strong>{group.label}</strong>
      {group.description ? <p className="walkthrough-muted">{group.description}</p> : null}
      <ParameterTable params={group.parameters} />
    </div>
  );
}

// JSON artifacts often wrap their real payload under one key (e.g. feature_vector.json is
// {"features": {...}}) -- unwrap that one level so the table shows the actual fields instead of a
// single "features: 34 fields" row.
function jsonPreviewEntries(data: unknown): Array<[string, unknown]> | null {
  if (!data || typeof data !== "object" || Array.isArray(data)) return null;
  const obj = data as Record<string, unknown>;
  const keys = Object.keys(obj);
  if (keys.length === 1) {
    const only = obj[keys[0]];
    if (only && typeof only === "object" && !Array.isArray(only) && Object.keys(only as object).length > 0) {
      return jsonPreviewEntries(only);
    }
  }
  return keys.map((key) => [key, obj[key]]);
}

function formatJsonPreviewValue(value: unknown): string {
  if (value === null || value === undefined) return "—";
  if (typeof value === "number") return String(Math.round(value * 10000) / 10000);
  if (typeof value === "boolean") return value ? "true" : "false";
  if (Array.isArray(value)) {
    if (value.length === 0) return "none";
    if (value.every((item) => item === null || typeof item !== "object")) return value.map(formatJsonPreviewValue).join(", ");
    return `${value.length} item${value.length === 1 ? "" : "s"}`;
  }
  if (typeof value === "object") {
    const keys = Object.keys(value as Record<string, unknown>);
    return keys.length === 0 ? "none" : `${keys.length} field${keys.length === 1 ? "" : "s"}`;
  }
  return String(value);
}

function JsonArtifactResult({ takeId, artifact }: { takeId: string; artifact: StudioArtifact }) {
  const [data, setData] = useState<unknown>(undefined);

  useEffect(() => {
    let cancelled = false;
    setData(undefined);
    if (!artifact.path) {
      setData(null);
      return;
    }
    fetch(fileUrl(takeId, artifact.path))
      .then((response) => {
        if (!response.ok) throw new Error(String(response.status));
        return response.json();
      })
      .then((json) => {
        if (!cancelled) setData(json);
      })
      .catch(() => {
        if (!cancelled) setData(null);
      });
    return () => {
      cancelled = true;
    };
  }, [takeId, artifact.path]);

  if (data === undefined) return <small className="walkthrough-muted">Loading…</small>;
  if (data === null) return <small className="walkthrough-muted">Could not load this file.</small>;
  const entries = jsonPreviewEntries(data);
  if (!entries || entries.length === 0) return <small className="walkthrough-muted">Empty.</small>;
  return (
    <>
      <table className="walkthrough-json-result-table">
        <tbody>
          {entries.map(([key, value]) => (
            <tr key={key}>
              <td>{key.replace(/_/g, " ")}</td>
              <td>{formatJsonPreviewValue(value)}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <details className="walkthrough-json-raw">
        <summary>Raw JSON</summary>
        <pre>{JSON.stringify(data, null, 2)}</pre>
      </details>
    </>
  );
}

function SubstageBlock({
  takeId,
  pipelineId,
  substage,
  variant = "substage",
  onOpenImage,
}: {
  takeId: string;
  pipelineId: string;
  substage: WalkthroughSubstage;
  variant?: "overview" | "substage";
  onOpenImage: (state: LightboxState) => void;
}) {
  const isOverview = variant === "overview";
  const isDeemphasized = substage.relevance === "inactive_strategy" || substage.relevance === "legacy";
  const hasParams = substage.parameterGroups.length > 0 || substage.ungroupedParameters.length > 0;
  const tuneHref = buildStudioDeepLink({ take_id: takeId, pipeline_id: pipelineId, unit_id: substage.unitId, tab: "tune" });
  const HeadingTag = isOverview ? "h3" : "h4";
  // JSON-kind artifacts never had a thumbnail in the image grid (just an empty box) -- pull them
  // into their own "Results" section instead, where their actual computed values can be shown.
  const imageArtifacts = substage.artifacts.filter((entry) => entry.kind !== "json");
  const jsonArtifacts = substage.artifacts.filter((entry) => entry.kind === "json");
  return (
    <div className={`walkthrough-substage${isOverview ? " overview" : ""}${isDeemphasized ? " deemphasized" : ""}`}>
      <header>
        {isOverview ? <span className="walkthrough-substage-kicker">Stage overview</span> : null}
        <HeadingTag>
          {substage.label}
          <RelevanceBadge relevance={substage.relevance} />
        </HeadingTag>
        <a className="walkthrough-tune-link" href={tuneHref} title="Open this step in the Tune panel">Tune this step →</a>
        {isDeemphasized ? <small className="walkthrough-muted">Not used by the current run's strategy.</small> : null}
        {substage.description ? <p className="walkthrough-muted">{substage.description}</p> : null}
      </header>

      {imageArtifacts.length > 0 ? (
        <div className="walkthrough-artifact-grid">
          {imageArtifacts.map((entry) => {
            const artifactTuneHref = buildStudioDeepLink({
              take_id: takeId,
              pipeline_id: pipelineId,
              unit_id: entry.provenance?.tuneUnitId ?? substage.unitId,
              tab: "tune",
            });
            return <ArtifactChip key={entry.artifactId} takeId={takeId} entry={entry} tuneHref={artifactTuneHref} onOpen={onOpenImage} />;
          })}
        </div>
      ) : null}

      {jsonArtifacts.length > 0 ? (
        <div className="walkthrough-json-results" aria-label={`${substage.label} results`}>
          <p className="walkthrough-substage-group-label">Results</p>
          {jsonArtifacts.map((entry) => {
            const jsonDeemphasized = entry.relevance === "inactive_strategy" || entry.relevance === "legacy";
            return (
              <div key={entry.artifactId} className={`walkthrough-json-result${jsonDeemphasized ? " deemphasized" : ""}`} title={entry.reason ?? undefined}>
                <div className="walkthrough-json-result-label">
                  <span>{entry.label}</span>
                  <IoRoleTag ioRole={entry.ioRole} feedsInto={entry.feedsInto} />
                  <RelevanceBadge relevance={entry.relevance} feedsInto={entry.feedsInto} />
                </div>
                {entry.artifact ? <JsonArtifactResult takeId={takeId} artifact={entry.artifact} /> : <small className="walkthrough-muted">Not produced for this run.</small>}
              </div>
            );
          })}
        </div>
      ) : null}

      {hasParams ? (
        <div className="walkthrough-parameters">
          {substage.parameterGroups.map((group) => <ParameterGroupBlock key={group.id} group={group} />)}
          {substage.ungroupedParameters.length > 0 ? (
            <div className="walkthrough-parameter-group">
              {substage.parameterGroups.length > 0 ? <strong>Other parameters</strong> : null}
              <ParameterTable params={substage.ungroupedParameters} />
            </div>
          ) : null}
        </div>
      ) : (
        <p className="walkthrough-muted">No tunable parameters on this step.</p>
      )}

      {substage.downstreamEffects.length > 0 ? (
        <details className="walkthrough-effects">
          <summary>Downstream effects</summary>
          <ul>{substage.downstreamEffects.map((effect) => <li key={effect}>{effect}</li>)}</ul>
        </details>
      ) : null}

      {substage.tuningHints.length > 0 ? (
        <details className="walkthrough-tuning-hints">
          <summary>What to tune</summary>
          <ul>
            {substage.tuningHints.map((hint) => (
              <li key={hint.condition}>
                <strong>{hint.condition}:</strong> {hint.actions.join(" • ")}
              </li>
            ))}
          </ul>
        </details>
      ) : null}
    </div>
  );
}

function StageSection({
  takeId,
  pipelineId,
  stage,
  collapsed,
  onToggleCollapsed,
  onOpenImage,
}: {
  takeId: string;
  pipelineId: string;
  stage: WalkthroughStage;
  collapsed: boolean;
  onToggleCollapsed: () => void;
  onOpenImage: (state: LightboxState) => void;
}) {
  if (stage.substages.length === 0) return null;
  const overview = stage.substages.find((substage) => substage.substageId === "strategy_path") ?? null;
  const childSubstages = stage.substages.filter((substage) => substage.substageId !== "strategy_path");
  return (
    <section className={`walkthrough-stage${collapsed ? " collapsed" : ""}`}>
      <button type="button" className="walkthrough-stage-header" onClick={onToggleCollapsed} aria-expanded={!collapsed}>
        <span className={`walkthrough-stage-chevron${collapsed ? " collapsed" : ""}`} aria-hidden>▾</span>
        <h2>{stage.label}</h2>
      </button>
      {stage.description ? <p className="walkthrough-muted">{stage.description}</p> : null}
      {stage.outputs.length > 0 ? (
        <div className="walkthrough-stage-outputs" aria-label="This stage's output(s)">
          <span className="walkthrough-stage-outputs-label">{stage.outputs.length > 1 ? "Outputs" : "Output"}</span>
          {stage.outputs.map((output) => (
            <span
              key={output.id}
              className={`walkthrough-stage-output-chip${output.present ? "" : " missing"}`}
              title={output.present ? output.label : `${output.label} — not produced for this run`}
            >
              {output.artifact?.title ?? output.label}
            </span>
          ))}
        </div>
      ) : null}
      {stage.strategySummary?.length ? (
        <div className="walkthrough-strategy-summary" aria-label="Selected strategy for this run">
          {stage.strategySummary.map((entry) => (
            <div key={entry.label}>
              <span>{entry.label}</span>
              <strong>{entry.value}</strong>
            </div>
          ))}
        </div>
      ) : null}
      {overview ? (
        <SubstageBlock
          key={overview.unitId}
          takeId={takeId}
          pipelineId={pipelineId}
          substage={overview}
          variant="overview"
          onOpenImage={onOpenImage}
        />
      ) : null}
      <div className={`walkthrough-stage-body${collapsed ? " collapsed" : ""}`}>
        {childSubstages.length > 0 ? (
          <div className="walkthrough-substage-group" aria-label={`${stage.label} substeps`}>
            <p className="walkthrough-substage-group-label">Substeps</p>
            {childSubstages.map((substage) => (
              <SubstageBlock key={substage.unitId} takeId={takeId} pipelineId={pipelineId} substage={substage} onOpenImage={onOpenImage} />
            ))}
          </div>
        ) : null}
      </div>
    </section>
  );
}

export default function PipelineWalkthroughPage() {
  const params = queryParams();
  const takeId = params.get("take_id") ?? "";
  const pipelineId = params.get("pipeline_id") ?? "";
  const [pipeline, setPipeline] = useState<PipelineInfo | null>(null);
  const [detail, setDetail] = useState<TakeDetail | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [lightbox, setLightbox] = useState<LightboxState>(null);
  const [collapsedStageIds, setCollapsedStageIds] = useState<Set<string>>(new Set());

  const load = useCallback(async () => {
    if (!takeId || !pipelineId) {
      setError("This report needs both take_id and pipeline_id in the URL.");
      setLoading(false);
      return;
    }
    setLoading(true);
    try {
      const [pipelines, takeDetail] = await Promise.all([api.pipelines(), api.take(takeId)]);
      const matched = pipelines.find((item) => item.id === pipelineId) ?? null;
      setPipeline(matched);
      setDetail(takeDetail);
      setError(matched ? null : `Pipeline "${pipelineId}" was not found.`);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to load take/pipeline data");
    } finally {
      setLoading(false);
    }
  }, [takeId, pipelineId]);

  useEffect(() => {
    void load();
  }, [load]);

  // CSS alone can't reliably force a native <details> open across browsers, so printing (which
  // otherwise shows the full report regardless of on-screen collapse state) needs a direct nudge
  // for the "Downstream effects" / "What to tune" / "Raw JSON" disclosures specifically.
  useEffect(() => {
    let openedByPrint: HTMLDetailsElement[] = [];
    const handleBeforePrint = () => {
      const all = Array.from(document.querySelectorAll<HTMLDetailsElement>("details.walkthrough-effects, details.walkthrough-tuning-hints, details.walkthrough-json-raw"));
      openedByPrint = all.filter((el) => !el.open);
      openedByPrint.forEach((el) => { el.open = true; });
    };
    const handleAfterPrint = () => {
      openedByPrint.forEach((el) => { el.open = false; });
      openedByPrint = [];
    };
    window.addEventListener("beforeprint", handleBeforePrint);
    window.addEventListener("afterprint", handleAfterPrint);
    return () => {
      window.removeEventListener("beforeprint", handleBeforePrint);
      window.removeEventListener("afterprint", handleAfterPrint);
    };
  }, []);

  if (loading) return <main className="page-pad"><div className="empty-state">Loading walkthrough…</div></main>;
  if (error) return <main className="page-pad"><div className="empty-state">{error}</div></main>;

  const stages = buildPipelineWalkthrough(pipeline, detail);
  const allCollapsed = stages.length > 0 && stages.every((stage) => collapsedStageIds.has(stage.stageId));
  const toggleSummaryMode = () => {
    setCollapsedStageIds(allCollapsed ? new Set() : new Set(stages.map((stage) => stage.stageId)));
  };
  const toggleStageCollapsed = (stageId: string) => {
    setCollapsedStageIds((prev) => {
      const next = new Set(prev);
      if (next.has(stageId)) next.delete(stageId);
      else next.add(stageId);
      return next;
    });
  };

  return (
    <main className="page-pad walkthrough-page">
      <header className="walkthrough-header">
        <div>
          <h1>Pipeline walkthrough</h1>
          <p className="walkthrough-muted">
            Take <strong>{takeId}</strong> · Pipeline <strong>{pipeline?.display_name ?? pipelineId}</strong>
          </p>
        </div>
        <div className="walkthrough-header-actions">
          <button type="button" onClick={toggleSummaryMode}>{allCollapsed ? "Expand all" : "Summary view"}</button>
          <a href={`/studio?take_id=${encodeURIComponent(takeId)}&pipeline_id=${encodeURIComponent(pipelineId)}`}>Back to Studio</a>
          <button type="button" onClick={() => window.print()}>Print</button>
        </div>
      </header>
      <p className="walkthrough-legend">
        Steps and parameters that did not run for this particular take (a different strategy branch, or diagnostics-only
        output) are shown dimmed, with a badge explaining why — nothing is hidden, so this stays complete as a printed
        reference. Click a thumbnail to view it full-size, or "Tune this step" to jump into the live editor. Click a
        stage's header, or "Summary view", to collapse it down to just the header.
      </p>
      {stages.map((stage) => (
        <StageSection
          key={stage.stageId}
          takeId={takeId}
          pipelineId={pipelineId}
          stage={stage}
          collapsed={collapsedStageIds.has(stage.stageId)}
          onToggleCollapsed={() => toggleStageCollapsed(stage.stageId)}
          onOpenImage={setLightbox}
        />
      ))}
      <Lightbox detail={detail} state={lightbox} takeId={takeId} onClose={() => setLightbox(null)} />
    </main>
  );
}
