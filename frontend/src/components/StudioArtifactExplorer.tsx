import ProjectionArtifactViewer from "./ProjectionArtifactViewer";
import { fileUrl } from "../api/client";
import { useEffect, useMemo, useState } from "react";
import { overlayDebugInfo, resolveOverlayTarget, visibleOverlays, type OverlayDebugInfo, type OverlayDisplayMode } from "./overlayModel";
import { type StudioArtifact } from "./studioWorkspaceModel";
import { objectDisplayId } from "./objectSelectionModel";
import type { RoiPolygon, RoiRect } from "./roiMapping";

type Props = {
  artifacts: StudioArtifact[];
  takeId?: string | null;
  selectedArtifactId: string | null;
  onSelect: (artifactId: string) => void;
  detailJson?: unknown;
  selectedObjectId?: number | null;
  hoveredObjectId?: number | null;
  hoveredArtifactId?: string | null;
  onSelectObject?: (objectId: number) => void;
  onHoverObject?: (objectId: number | null) => void;
  onHoverArtifact?: (artifactId: string | null) => void;
  onOverlayDebug?: (info: OverlayDebugInfo) => void;
  roi?: RoiRect | null;
  roiPolygon?: RoiPolygon | null;
  roiType?: "rectangle" | "polygon";
  roiEnabled?: boolean;
  onApplyRoi?: (roi: RoiRect, rerun: boolean) => void;
  onApplyPolygonRoi?: (points: RoiPolygon, rerun: boolean) => void;
  onClearRoi?: () => void;
};

export default function StudioArtifactExplorer({
  artifacts,
  takeId,
  selectedArtifactId,
  onSelect,
  detailJson,
  selectedObjectId,
  hoveredObjectId = null,
  hoveredArtifactId = null,
  onSelectObject,
  onHoverObject,
  onHoverArtifact,
  onOverlayDebug,
  roi = null,
  roiPolygon = null,
  roiType = "rectangle",
  roiEnabled = false,
  onApplyRoi,
  onApplyPolygonRoi,
  onClearRoi,
}: Props) {
  const [overlayMode, setOverlayMode] = useState<OverlayDisplayMode>("all");
  const [drawRoiMode, setDrawRoiMode] = useState(false);
  const [draftRoi, setDraftRoi] = useState<RoiRect | null>(null);
  const [draftPolygon, setDraftPolygon] = useState<RoiPolygon | null>(null);
  const [localRoiType, setLocalRoiType] = useState<"rectangle" | "polygon">(roiType);
  useEffect(() => {
    setLocalRoiType(roiType);
  }, [roiType]);
  if (!artifacts.length) {
    return <div className="empty-state">No artifacts generated for this context yet.</div>;
  }
  const selected = artifacts.find((artifact) => artifact.artifact_id === selectedArtifactId) ?? artifacts[0];
  const selectedOverlayDebug = selected.kind === "overlay" ? overlayDebugInfo(selected, artifacts, { width: 960, height: 540 }) : null;
  const targetResolution = resolveOverlayTarget(artifacts, selected);
  const previewTarget = targetResolution.target;
  const selectedOverlayId = selected.kind === "overlay" ? selected.artifact_id : null;
  const morphDebug = artifacts.find((item) => item.artifact_id === "morphology_debug_json");
  const morphMetrics = artifacts.find((item) => item.artifact_id === "morphology_metrics");
  const debugMeta = (morphDebug?.metadata ?? {}) as Record<string, unknown>;
  const metricsMeta = (morphMetrics?.metadata ?? {}) as Record<string, unknown>;
  const morph = (debugMeta.morphology as Record<string, unknown> | undefined) ?? {};
  const threshold = (debugMeta.threshold as Record<string, unknown> | undefined) ?? {};
  const roiInfo = (debugMeta.roi as Record<string, unknown> | undefined) ?? {};
  const components = Number(morph.components_after ?? metricsMeta.connected_components ?? 0);
  const cleanedCoverage = Number(morph.cleaned_foreground_coverage ?? metricsMeta.cleaned_foreground_coverage ?? 0);
  const thresholdCoverage = Number(threshold.foreground_coverage ?? metricsMeta.threshold_foreground_coverage ?? 0);
  const emptyCleaned = Number(morph.cleaned_foreground_pixels ?? metricsMeta.cleaned_foreground_pixels ?? 0) <= 0;
  const overlays = useMemo(
    () => visibleOverlays(targetResolution.overlays, overlayMode, selectedObjectId ?? null),
    [overlayMode, selectedObjectId, targetResolution.overlays]
  );
  return (
    <div className="studio-artifact-explorer">
      <aside className="studio-artifact-nav">
        {artifacts.map((artifact) => (
          <button
            className={[
              artifact.artifact_id === selected.artifact_id ? "active" : "",
              hoveredArtifactId === artifact.artifact_id ? "overlay-linked-hover" : "",
            ].join(" ")}
            key={artifact.artifact_id}
            onClick={() => onSelect(artifact.artifact_id)}
            onMouseEnter={() => {
              onHoverObject?.(artifact.object_id ?? null);
              onHoverArtifact?.(artifact.artifact_id);
            }}
            onMouseLeave={() => {
              onHoverObject?.(null);
              onHoverArtifact?.(null);
            }}
            type="button"
          >
            <strong>{artifact.title}</strong>
            <span>{artifact.stage_id}</span>
            <small>{artifact.kind}</small>
            <small>{artifact.object_id == null ? "global" : objectDisplayId(artifact.object_id)}</small>
          </button>
        ))}
      </aside>
      <section className="studio-artifact-preview">
        {(selected.kind === "overlay" || targetResolution.overlays.length > 0) && (
          <div className="overlay-controls">
            <button className={overlayMode === "all" ? "active" : ""} onClick={() => setOverlayMode("all")} type="button">Show all overlays</button>
            <button className={overlayMode === "selected_only" ? "active" : ""} onClick={() => setOverlayMode("selected_only")} type="button">Selected object only</button>
            <button className={overlayMode === "hide" ? "active" : ""} onClick={() => setOverlayMode("hide")} type="button">Hide overlays</button>
          </div>
        )}
        {!!targetResolution.warning && (
          <div className="artifact-warning-banner">
            {targetResolution.warning}
          </div>
        )}
        {!!selectedOverlayDebug?.warnings.length && (
          <div className="artifact-warning-banner">
            {selectedOverlayDebug.warnings.join(" | ")}
          </div>
        )}
        {selectedOverlayDebug?.approximate && (
          <div className="artifact-warning-banner approximate">
            APPROXIMATE
          </div>
        )}
        {(selected.kind === "image" || selected.kind === "overlay") && previewTarget?.path && takeId && (
          <ProjectionArtifactViewer
            title={previewTarget.title}
            src={fileUrl(takeId, previewTarget.path)}
            overlays={overlays}
            selectedObjectId={selectedObjectId ?? null}
            hoveredObjectId={hoveredObjectId}
            selectedOverlayArtifactId={selectedOverlayId}
            displayMode={overlayMode}
            onSelectObject={onSelectObject}
            onSelectOverlayArtifact={onSelect}
            onHoverObject={onHoverObject}
            onHoverOverlayArtifact={onHoverArtifact}
            onOverlayDebug={onOverlayDebug}
            artifacts={artifacts}
            emptyMessage="Artifact file is unavailable."
            roi={localRoiType === "rectangle" ? (draftRoi ?? (roiEnabled && roiType === "rectangle" ? roi : null)) : null}
            roiPolygon={localRoiType === "polygon" ? (draftPolygon ?? (roiEnabled && roiType === "polygon" ? roiPolygon : null)) : null}
            roiType={localRoiType}
            drawRoiMode={drawRoiMode}
            onRoiDrawComplete={(next) => setDraftRoi(next)}
            onPolygonRoiComplete={(points) => setDraftPolygon(points)}
          />
        )}
        {selected.stage_id === "segmentation" && (
          <div className="pipeline-status-grid">
            <div><span>Components</span><strong>{Number.isFinite(components) ? components : "-"}</strong></div>
            <div><span>Threshold coverage</span><strong>{Number.isFinite(thresholdCoverage) ? `${(thresholdCoverage * 100).toFixed(1)}%` : "-"}</strong></div>
            <div><span>Cleaned coverage</span><strong>{Number.isFinite(cleanedCoverage) ? `${(cleanedCoverage * 100).toFixed(1)}%` : "-"}</strong></div>
            <div><span>ROI</span><strong>{Boolean(roiInfo.enabled ?? roiEnabled) ? "enabled" : "disabled"}</strong></div>
          </div>
        )}
        {selected.stage_id === "segmentation" && (
          <div className="overlay-controls">
            <button className={roiEnabled ? "active" : ""} type="button">{`ROI: ${roiEnabled ? "enabled" : "disabled"}`}</button>
            <button className={localRoiType === "rectangle" ? "active" : ""} onClick={() => setLocalRoiType("rectangle")} type="button">Rectangle</button>
            <button className={localRoiType === "polygon" ? "active" : ""} onClick={() => setLocalRoiType("polygon")} type="button">Polygon</button>
            <button className={drawRoiMode ? "active" : ""} onClick={() => setDrawRoiMode((value) => !value)} type="button">Draw ROI</button>
            <button onClick={() => { setDraftRoi(null); setDraftPolygon(null); onClearRoi?.(); }} type="button">Clear ROI</button>
            <button
              disabled={localRoiType === "rectangle" ? !draftRoi : !(draftPolygon && draftPolygon.length >= 3)}
              onClick={() => {
                if (localRoiType === "rectangle" && draftRoi) onApplyRoi?.(draftRoi, false);
                if (localRoiType === "polygon" && draftPolygon && draftPolygon.length >= 3) onApplyPolygonRoi?.(draftPolygon, false);
              }}
              type="button"
            >
              Apply ROI
            </button>
            <button
              disabled={localRoiType === "rectangle" ? !draftRoi : !(draftPolygon && draftPolygon.length >= 3)}
              onClick={() => {
                if (localRoiType === "rectangle" && draftRoi) onApplyRoi?.(draftRoi, true);
                if (localRoiType === "polygon" && draftPolygon && draftPolygon.length >= 3) onApplyPolygonRoi?.(draftPolygon, true);
              }}
              type="button"
            >
              Apply ROI + rerun
            </button>
          </div>
        )}
        {selected.stage_id === "segmentation" && emptyCleaned && (
          <div className="artifact-warning-banner">
            Cleaned mask is empty. Try lowering min component area, changing threshold/invert, or selecting a tighter ROI.
          </div>
        )}
        <details className="artifact-reference">
          <summary>Artifact details</summary>
          <small>Stage: {selected.stage_id}</small>
          <small>Kind: {selected.kind}</small>
          <small>Generated: {selected.generated ?? "derived"}</small>
          <small>Object: {selected.object_id == null ? "-" : objectDisplayId(selected.object_id)}</small>
          <small>Preview: {selected.preview_available ? "available" : "not available"}</small>
          <small>File: {selected.path ?? "-"}</small>
          <small>Focused object: {selectedObjectId == null ? "-" : selectedObjectId}</small>
          <small>Target image: {previewTarget ? `${previewTarget.title} (${previewTarget.artifact_id})` : "-"}</small>
          <small>Coordinate space: {selected.coordinate_space ?? "-"}</small>
          <small>Overlay type: {selected.overlay_type ?? "-"}</small>
          <small>Projection type: {previewTarget?.projection_type ?? "-"}</small>
          <small>Projection transform: {previewTarget?.projection_transform_id ?? "-"}</small>
          <small>Source artifact: {String(selected.metadata?.source_artifact_id ?? "-")}</small>
          <small>Mask artifact: {String(selected.metadata?.mask_artifact_id ?? "-")}</small>
        </details>
        {selected.kind === "json" && <pre className="json-block">{JSON.stringify(detailJson ?? {}, null, 2)}</pre>}
        {selected.kind === "table" && <div className="artifact-reference"><strong>Table artifact ready for stage/object inspection.</strong></div>}
        {selected.kind === "metric" && <div className="artifact-reference"><strong>Metric artifact ready for stage/object inspection.</strong></div>}
        {(selected.kind === "point_cloud" || selected.kind === "file" || selected.kind === "video" || selected.kind === "text") && (
          <div className="empty-state">Preview not available for this artifact kind yet.</div>
        )}
      </section>
    </div>
  );
}
