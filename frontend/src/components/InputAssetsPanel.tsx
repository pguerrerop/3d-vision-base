import { useMemo, useState } from "react";
import { fileUrl, type CaptureModality, type ProcessingResult, type TakeDetail } from "../api/client";
import ImagePanel from "./ImagePanel";
import { availableInputTabs } from "./inputAssetsModel";
import PointCloudStatsCard from "./PointCloudStatsCard";

type Props = {
  detail: TakeDetail | null;
  result?: ProcessingResult | null;
  compact?: boolean;
};

const LABELS: Record<CaptureModality | "metadata", string> = {
  point_cloud: "Point cloud",
  heightmap: "Heightmap (Primary)",
  reflectance: "Reflectance",
  rgb: "RGB",
  rgb_video: "RGB video",
  laser_rgb: "Laser image",
  metadata: "Metadata"
};

const IMAGE_KEYS = ["heightmap", "height", "reflectance", "rgb", "laser_rgb", "laser_overlay", "laser_image"];

function JsonBlock({ value }: { value: unknown }) {
  return <pre className="json-block">{JSON.stringify(value ?? {}, null, 2)}</pre>;
}

export default function InputAssetsPanel({ detail, result, compact = false }: Props) {
  const tabs = useMemo(() => availableInputTabs(detail), [detail]);
  const [active, setActive] = useState<CaptureModality | "metadata">(tabs[0] ?? "metadata");
  const selected = tabs.includes(active) ? active : tabs[0] ?? "metadata";
  const assets = detail?.assets ?? {};
  const waitingMessage = detail?.has_ready && !detail?.has_done ? "This take has not been processed yet." : "No asset available for this modality.";

  if (!detail) {
    return <div className="empty-state">Select a take to inspect inputs.</div>;
  }

  return (
    <section className={compact ? "input-assets-panel compact" : "input-assets-panel"}>
      <div className="asset-tabs">
        {tabs.map((tab) => (
          <button className={tab === selected ? "active" : ""} key={tab} onClick={() => setActive(tab)} type="button">
            {LABELS[tab]}
          </button>
        ))}
      </div>
      {selected === "point_cloud" && (
        <div className="raw-input-grid">
          <ImagePanel title="Point cloud preview" src={result?.files.input_preview ? fileUrl(detail.take_id, result.files.input_preview) : null} emptyMessage={waitingMessage} />
          <PointCloudStatsCard stats={result?.input_stats} />
        </div>
      )}
      {selected !== "point_cloud" && selected !== "metadata" && selected === "rgb_video" && (
        <VideoAsset detail={detail} assetGroup={assets[selected]} />
      )}
      {selected !== "point_cloud" && selected !== "metadata" && selected !== "rgb_video" && (
        <ImagePanel
          title={LABELS[selected]}
          src={firstAssetUrl(detail.take_id, assets[selected])}
          emptyMessage={`No ${LABELS[selected].toLowerCase()} asset is available for this take.`}
        />
      )}
      {selected === "metadata" && <JsonBlock value={{ modalities: detail.modalities, frameset: detail.frameset, assets: detail.assets, metadata: detail.metadata }} />}
    </section>
  );
}

function firstAssetUrl(takeId: string, assetGroup: Record<string, string> | undefined): string | null {
  if (!assetGroup) return null;
  if (assetGroup.heightmap_preview) return fileUrl(takeId, assetGroup.heightmap_preview);
  for (const key of IMAGE_KEYS) {
    const filename = assetGroup[key];
    if (filename) return fileUrl(takeId, filename);
  }
  const first = Object.values(assetGroup)[0];
  return first ? fileUrl(takeId, first) : null;
}

function VideoAsset({ detail, assetGroup }: { detail: TakeDetail; assetGroup: Record<string, string> | undefined }) {
  const filename = assetGroup?.rgb_video ?? Object.values(assetGroup ?? {})[0];
  const videoUrl = filename ? fileUrl(detail.take_id, filename) : null;
  const previewUrl = firstAssetUrl(detail.take_id, detail.assets?.rgb);
  const source = typeof detail.metadata?.source === "object" && detail.metadata.source ? detail.metadata.source as Record<string, unknown> : {};
  const frameCount = detail.frame_count ?? (typeof detail.metadata?.frame_count === "number" ? detail.metadata.frame_count : null);

  return (
    <div className="video-asset">
      {previewUrl && <ImagePanel title="Video preview" src={previewUrl} emptyMessage="No preview image is available for this video." />}
      <div className="video-asset-info">
        <span>File</span>
        <strong>{filename ?? "-"}</strong>
        <span>Frames</span>
        <strong>{frameCount ?? "-"}</strong>
        <span>FPS</span>
        <strong>{typeof source.fps === "number" ? source.fps.toFixed(2) : "-"}</strong>
        {videoUrl && (
          <a href={videoUrl} target="_blank" rel="noreferrer">
            Open video
          </a>
        )}
      </div>
    </div>
  );
}
