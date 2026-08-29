import { useEffect, useMemo, useRef, useState } from "react";
import { fileUrl } from "../api/client";
import type { TakeDetail } from "../api/client";
import type { StudioArtifact } from "./studioWorkspaceModel";
import { primarySourceImageUrl } from "./stage_sources";
import {
  clusterColor,
  hslToRgb,
  loadImageElement,
  memberIdsForClusterPreview,
  resolveBlobIdMaskArtifact,
} from "./clusterPreviewModel";
import type { ClusterRow } from "./clusterDecisionModel";

function resolveClusterPreviewBaseUrl(
  takeId: string,
  detail: TakeDetail | null,
  stageArtifacts: StudioArtifact[],
): string | null {
  const stagePreview = stageArtifacts.find((artifact) =>
    artifact.path
    && (artifact.artifact_id === "raw_heightmap_preview"
      || artifact.artifact_id === "heightmap_input_preview"
      || artifact.artifact_id === "input_preview"));
  if (stagePreview?.path) return fileUrl(takeId, stagePreview.path);
  return primarySourceImageUrl(detail);
}

type Props = {
  takeId: string;
  detail: TakeDetail | null;
  artifacts: StudioArtifact[];
  activeRow: ClusterRow | null;
};

function renderClusterPreview(
  canvas: HTMLCanvasElement,
  baseImage: HTMLImageElement,
  maskImage: HTMLImageElement,
  memberIds: Set<number>,
  tint: [number, number, number],
): void {
  const width = baseImage.naturalWidth || baseImage.width;
  const height = baseImage.naturalHeight || baseImage.height;
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d");
  if (!ctx) return;

  const maskCanvas = document.createElement("canvas");
  maskCanvas.width = width;
  maskCanvas.height = height;
  const maskCtx = maskCanvas.getContext("2d");
  if (!maskCtx) return;

  ctx.drawImage(baseImage, 0, 0, width, height);
  maskCtx.drawImage(maskImage, 0, 0, width, height);
  const base = ctx.getImageData(0, 0, width, height);
  const mask = maskCtx.getImageData(0, 0, width, height);
  const [tr, tg, tb] = tint;

  for (let i = 0; i < base.data.length; i += 4) {
    const label = mask.data[i];
    if (memberIds.has(label)) {
      base.data[i] = Math.round(base.data[i] * 0.35 + tr * 0.65);
      base.data[i + 1] = Math.round(base.data[i + 1] * 0.35 + tg * 0.65);
      base.data[i + 2] = Math.round(base.data[i + 2] * 0.35 + tb * 0.65);
    } else {
      base.data[i] = Math.round(base.data[i] * 0.22);
      base.data[i + 1] = Math.round(base.data[i + 1] * 0.22);
      base.data[i + 2] = Math.round(base.data[i + 2] * 0.22);
    }
    base.data[i + 3] = 255;
  }

  ctx.putImageData(base, 0, 0);
}

export default function BlobClusterPreview({ takeId, detail, artifacts, activeRow }: Props) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const maskArtifact = useMemo(() => resolveBlobIdMaskArtifact(artifacts), [artifacts]);
  const baseUrl = useMemo(() => resolveClusterPreviewBaseUrl(takeId, detail, artifacts), [takeId, detail, artifacts]);
  const memberIds = useMemo(() => {
    if (!activeRow) return [];
    return memberIdsForClusterPreview(activeRow, maskArtifact);
  }, [activeRow, maskArtifact]);

  useEffect(() => {
    const canvas = canvasRef.current;
    const maskPath = maskArtifact?.path ?? null;
    const clusterId = activeRow?.clusterId ?? null;
    if (!canvas || !clusterId || !baseUrl || !maskPath || !memberIds.length) {
      setError(null);
      setLoading(false);
      return;
    }
    const resolvedBaseUrl = baseUrl;
    const resolvedMaskPath = maskPath;
    const resolvedClusterId = clusterId;

    let cancelled = false;
    setLoading(true);
    setError(null);

    async function drawPreview() {
      try {
        const [baseImage, maskImage] = await Promise.all([
          loadImageElement(resolvedBaseUrl),
          loadImageElement(fileUrl(takeId, resolvedMaskPath)),
        ]);
        if (cancelled || !canvasRef.current) return;
        renderClusterPreview(
          canvasRef.current,
          baseImage,
          maskImage,
          new Set(memberIds),
          hslToRgb(clusterColor(resolvedClusterId)),
        );
        setLoading(false);
      } catch (previewError) {
        if (cancelled) return;
        setLoading(false);
        setError(previewError instanceof Error ? previewError.message : "Failed to render cluster preview.");
      }
    }

    void drawPreview();
    return () => { cancelled = true; };
  }, [takeId, activeRow, baseUrl, maskArtifact?.path, memberIds]);

  if (!activeRow) {
    return (
      <div className="cluster-preview-panel empty-state">
        <strong>Select a cluster row to preview its spatial support.</strong>
        <small>Highlighted pixels match the cluster&apos;s fragment or component ids in the id mask.</small>
      </div>
    );
  }

  if (!maskArtifact?.path) {
    return (
      <div className="cluster-preview-panel empty-state">
        <strong>No blob id mask is available for this run.</strong>
        <small>Run component formation with the blob strategy to emit an id mask artifact.</small>
      </div>
    );
  }

  if (!baseUrl) {
    return (
      <div className="cluster-preview-panel empty-state">
        <strong>No reference image is available for cluster preview.</strong>
      </div>
    );
  }

  if (!memberIds.length) {
    return (
      <div className="cluster-preview-panel empty-state">
        <strong>{activeRow.clusterId} has no mapped fragment or component ids.</strong>
      </div>
    );
  }

  return (
    <div className="cluster-preview-panel">
      <div className="cluster-preview-header">
        <strong>
          <span className="cluster-chip" style={{ background: clusterColor(activeRow.clusterId) }} />
          {activeRow.clusterId} spatial preview
        </strong>
        <small>
          {memberIds.length} id{memberIds.length === 1 ? "" : "s"} highlighted
          {" · "}
          {activeRow.decision}
          {activeRow.rejectionReason ? ` · ${activeRow.rejectionReason}` : ""}
        </small>
      </div>
      <div className="cluster-preview-canvas-wrap">
        <canvas ref={canvasRef} className="cluster-preview-canvas" />
        {loading ? <div className="cluster-preview-loading">Rendering preview…</div> : null}
        {error ? <div className="cluster-preview-error">{error}</div> : null}
      </div>
    </div>
  );
}
