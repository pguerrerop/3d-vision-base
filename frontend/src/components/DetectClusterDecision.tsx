import { useEffect, useMemo, useState } from "react";
import { fileUrl } from "../api/client";
import type { TakeDetail } from "../api/client";
import BlobClusterPreview from "./BlobClusterPreview";
import type { StudioArtifact } from "./studioWorkspaceModel";
import { clusterColor } from "./clusterPreviewModel";
import {
  buildBridgeRows,
  magnitudeBucket,
  parseClusterDiagnostics,
  selectedClusterRow,
  topRejectedClusterRow,
  type ClusterDiagnostics,
  type ClusterRow,
} from "./clusterDecisionModel";

type Mode = "table" | "bridge" | "card";

type Props = {
  takeId: string;
  detail?: TakeDetail | null;
  artifacts: StudioArtifact[];
  mode: Mode;
  showSpatialPreview?: boolean;
  activeClusterId?: string | null;
  onActiveClusterChange?: (clusterId: string | null) => void;
};

function findArtifact(artifacts: StudioArtifact[], id: string): StudioArtifact | null {
  return artifacts.find((artifact) => artifact.artifact_id === id) ?? null;
}

async function fetchJson(takeId: string, path: string | null | undefined): Promise<unknown> {
  if (!path) return null;
  try {
    const response = await fetch(fileUrl(takeId, String(path)));
    if (!response.ok) return null;
    return await response.json();
  } catch {
    return null;
  }
}

function formatNumber(value: number | null, digits = 2): string {
  if (value == null) return "—";
  return Number.isInteger(value) ? String(value) : value.toFixed(digits);
}

function WhyCard({ diagnostics, activeRow }: { diagnostics: ClusterDiagnostics; activeRow: ClusterRow | null }) {
  const selected = activeRow ?? selectedClusterRow(diagnostics);
  const topRejected = topRejectedClusterRow(diagnostics);
  if (!selected) {
    return (
      <div className="cluster-why-card empty-state">
        <strong>No cluster-scoring diagnostics were emitted for this run.</strong>
        <small>Run the blob strategy and inspect the selected run rather than downstream object outputs.</small>
      </div>
    );
  }
  return (
    <div className="cluster-why-card">
      <div className="cluster-why-card-title">Why this cluster was selected</div>
      <div className="cluster-why-grid">
        <span>Selected cluster</span><strong style={{ color: clusterColor(selected.clusterId) }}>{selected.clusterId}</strong>
        <span>Decision</span><strong>{selected.decision}</strong>
        <span>Score</span><strong>{formatNumber(selected.score)}</strong>
        <span>Components/fragments</span><strong>{selected.componentCount} / {selected.fragmentCount}</strong>
        <span>Area</span><strong>{selected.areaFraction == null ? "—" : `${(selected.areaFraction * 100).toFixed(1)}% valid ROI`}</strong>
        <span>Median Z</span><strong>{selected.medianZ == null ? "—" : `${formatNumber(selected.medianZ)} mm`}</strong>
        <span>Z MAD</span><strong>{selected.zMad == null ? "—" : `${formatNumber(selected.zMad)} mm`}</strong>
        <span>Border contact</span><strong>{selected.borderContact == null ? "—" : (selected.borderContact ? "yes" : "no")}</strong>
        <span>Spatial coverage</span><strong>{magnitudeBucket(selected.spatialCoverage)}</strong>
        <span>Centrality penalty</span><strong>{magnitudeBucket(selected.centralityPenalty)}</strong>
        <span>Object-like penalty</span><strong>{magnitudeBucket(selected.objectLikePenalty)}</strong>
        <span>Stripe overlap</span><strong>{magnitudeBucket(selected.stripeOverlap)}</strong>
      </div>
      {topRejected ? (
        <div className="cluster-why-rejected">
          <span>Top rejected cluster:</span>
          <strong style={{ color: clusterColor(topRejected.clusterId) }}>{topRejected.clusterId}</strong>
          <small>Score: {formatNumber(topRejected.score)}</small>
          <small>Reason: {topRejected.rejectionReason ?? "lower weighted support score"}</small>
        </div>
      ) : null}
    </div>
  );
}

function DecisionTable({
  diagnostics,
  activeClusterId,
  onSelectCluster,
}: {
  diagnostics: ClusterDiagnostics;
  activeClusterId: string | null;
  onSelectCluster: (clusterId: string) => void;
}) {
  return (
    <div className="cluster-decision-table-wrap">
      <table className="cluster-decision-table">
        <thead>
          <tr>
            <th>selected</th>
            <th>cluster_id</th>
            <th>score</th>
            <th>decision</th>
            <th>rejection_reason</th>
            <th>comp/frag</th>
            <th>area_frac</th>
            <th>median_z</th>
            <th>z_mad</th>
            <th>border</th>
            <th>coverage</th>
            <th>centrality</th>
            <th>object-like</th>
            <th>fragmentation</th>
            <th>stripe</th>
          </tr>
        </thead>
        <tbody>
          {diagnostics.rows.map((row) => {
            const classes = [
              row.selected ? "is-selected" : "",
              row.clusterId === activeClusterId ? "is-active" : "",
            ].filter(Boolean).join(" ");
            return (
              <tr
                key={row.clusterId}
                className={classes}
                onClick={() => onSelectCluster(row.clusterId)}
                role="button"
                tabIndex={0}
                onKeyDown={(event) => { if (event.key === "Enter" || event.key === " ") onSelectCluster(row.clusterId); }}
              >
                <td>{row.selected ? "✓" : ""}</td>
                <td><span className="cluster-chip" style={{ background: clusterColor(row.clusterId) }} />{row.clusterId}</td>
                <td>{formatNumber(row.score)}</td>
                <td>{row.decision}</td>
                <td>{row.rejectionReason ?? "—"}</td>
                <td>{row.componentCount} / {row.fragmentCount}</td>
                <td>{formatNumber(row.areaFraction, 3)}</td>
                <td>{formatNumber(row.medianZ)}</td>
                <td>{formatNumber(row.zMad)}</td>
                <td>{row.borderContact == null ? "—" : (row.borderContact ? "yes" : "no")}</td>
                <td>{formatNumber(row.spatialCoverage)}</td>
                <td>{formatNumber(row.centralityPenalty)}</td>
                <td>{formatNumber(row.objectLikePenalty)}</td>
                <td>{formatNumber(row.fragmentationPenalty)}</td>
                <td>{formatNumber(row.stripeOverlap)}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function BridgeTable({
  diagnostics,
  activeClusterId,
  onSelectCluster,
}: {
  diagnostics: ClusterDiagnostics;
  activeClusterId: string | null;
  onSelectCluster: (clusterId: string) => void;
}) {
  const rows = useMemo(() => buildBridgeRows(diagnostics), [diagnostics]);
  return (
    <div className="cluster-decision-table-wrap">
      <table className="cluster-bridge-table">
        <thead>
          <tr>
            <th>Component / fragment</th>
            <th>Cluster</th>
            <th>Decision</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row, index) => (
            <tr
              key={`${row.clusterId}-${row.memberKind}-${row.memberId}-${index}`}
              className={[row.selected ? "is-selected" : "", row.clusterId === activeClusterId ? "is-active" : ""].filter(Boolean).join(" ")}
              onClick={() => onSelectCluster(row.clusterId)}
              role="button"
              tabIndex={0}
              onKeyDown={(event) => { if (event.key === "Enter" || event.key === " ") onSelectCluster(row.clusterId); }}
            >
              <td>{row.memberKind === "component" ? "component" : "fragment"}_{row.memberId}</td>
              <td><span className="cluster-chip" style={{ background: clusterColor(row.clusterId) }} />{row.clusterId}</td>
              <td>{row.decision === "selected" ? "selected" : (row.rejectionReason ? `rejected: ${row.rejectionReason}` : row.decision)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default function DetectClusterDecision({
  takeId,
  detail = null,
  artifacts,
  mode,
  showSpatialPreview = false,
  activeClusterId: controlledActiveClusterId,
  onActiveClusterChange,
}: Props) {
  const clustersArtifact = findArtifact(artifacts, "blob_height_clusters");
  const debugArtifact = findArtifact(artifacts, "blob_cluster_selection_debug");
  const [clustersJson, setClustersJson] = useState<unknown>(null);
  const [debugJson, setDebugJson] = useState<unknown>(null);
  const [loading, setLoading] = useState(true);
  const [internalActiveClusterId, setInternalActiveClusterId] = useState<string | null>(null);
  const isControlled = controlledActiveClusterId !== undefined;
  const activeClusterId = isControlled ? controlledActiveClusterId : internalActiveClusterId;
  const setActiveClusterId = (clusterId: string | null) => {
    if (!isControlled) setInternalActiveClusterId(clusterId);
    onActiveClusterChange?.(clusterId);
  };

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    async function load() {
      const [clusters, debug] = await Promise.all([
        fetchJson(takeId, clustersArtifact?.path),
        fetchJson(takeId, debugArtifact?.path),
      ]);
      if (cancelled) return;
      setClustersJson(clusters);
      setDebugJson(debug);
      setLoading(false);
    }
    void load();
    return () => { cancelled = true; };
  }, [takeId, clustersArtifact?.path, debugArtifact?.path]);

  const diagnostics = useMemo(() => parseClusterDiagnostics(clustersJson, debugJson), [clustersJson, debugJson]);
  const activeRow = useMemo(
    () => diagnostics.rows.find((row) => row.clusterId === activeClusterId) ?? selectedClusterRow(diagnostics),
    [diagnostics, activeClusterId],
  );

  if (loading) {
    return <div className="cluster-decision-panel"><small className="section-note">Loading cluster diagnostics…</small></div>;
  }

  if (!diagnostics.hasData) {
    return (
      <div className="cluster-decision-panel">
        <div className="empty-state">
          <strong>No cluster-scoring diagnostics were emitted for this run.</strong>
          <small>Run the blob strategy (low_gradient_blob_height_clusters) and inspect the selected run.</small>
        </div>
      </div>
    );
  }

  const effectiveActiveId = activeRow?.clusterId ?? null;
  const onSelectCluster = (clusterId: string) => setActiveClusterId(clusterId);

  return (
    <div className={`cluster-decision-with-preview${showSpatialPreview ? " is-split" : ""}`}>
      <div className="cluster-decision-panel">
        <WhyCard diagnostics={diagnostics} activeRow={activeRow} />
        {mode === "bridge" ? (
          <BridgeTable diagnostics={diagnostics} activeClusterId={effectiveActiveId} onSelectCluster={onSelectCluster} />
        ) : mode === "table" || mode === "card" ? (
          <DecisionTable diagnostics={diagnostics} activeClusterId={effectiveActiveId} onSelectCluster={onSelectCluster} />
        ) : null}
      </div>
      {showSpatialPreview ? (
        <BlobClusterPreview
          takeId={takeId}
          detail={detail}
          artifacts={artifacts}
          activeRow={activeRow}
        />
      ) : null}
    </div>
  );
}
