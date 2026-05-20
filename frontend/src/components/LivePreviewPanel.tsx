import { useEffect, useMemo, useState } from "react";
import { api, runtimePreviewUrl, type RuntimePreviewMetadata, type RuntimeState } from "../api/client";
import { imageAgeSeconds, livePreviewLabel, livePreviewStatus } from "./livePreviewModel";
import StatusBadge from "./StatusBadge";

type Props = {
  runtimeState: RuntimeState | null;
  compact?: boolean;
  refreshMs?: number;
  title?: string;
};

export default function LivePreviewPanel({ runtimeState, compact = false, refreshMs = 750, title = "Live camera preview" }: Props) {
  const [metadata, setMetadata] = useState<RuntimePreviewMetadata | null>(null);
  const [imageKey, setImageKey] = useState(Date.now());
  const [loadFailed, setLoadFailed] = useState(false);

  useEffect(() => {
    let canceled = false;
    async function refresh() {
      try {
        const next = await api.runtimePreviewMetadata();
        if (!canceled) {
          setMetadata(next);
          setImageKey(Date.now());
        }
      } catch {
        if (!canceled) {
          setMetadata(null);
          setLoadFailed(true);
        }
      }
    }
    void refresh();
    const timer = window.setInterval(() => void refresh(), refreshMs);
    return () => {
      canceled = true;
      window.clearInterval(timer);
    };
  }, [refreshMs]);

  const status = livePreviewStatus(runtimeState, metadata);
  const label = livePreviewLabel(status);
  const age = imageAgeSeconds(metadata?.timestamp);
  const resolution = metadata?.resolution?.length === 2 ? `${metadata.resolution[0]}x${metadata.resolution[1]}` : "-";
  const imageSrc = useMemo(() => runtimePreviewUrl(imageKey), [imageKey]);
  const shouldShowImage = status !== "unavailable" && !!metadata?.timestamp && !loadFailed;

  return (
    <section className={compact ? "live-preview-panel compact" : "live-preview-panel"}>
      <div className="live-preview-header">
        <div>
          <span className="eyebrow">{title}</span>
          <strong>{metadata?.source ?? runtimeState?.acquisition_source ?? "usb_camera"}</strong>
        </div>
        <StatusBadge value={label.toLowerCase()} />
      </div>
      <div className={`live-preview-frame ${status}`}>
        {shouldShowImage ? (
          <img src={imageSrc} alt="Latest live camera preview" onError={() => setLoadFailed(true)} onLoad={() => setLoadFailed(false)} />
        ) : (
          <div className="empty-image">{label}</div>
        )}
        {status !== "connected" && <div className="live-preview-overlay">{label}</div>}
      </div>
      <div className="live-preview-meta">
        <span>Camera {metadata?.camera_index ?? "-"}</span>
        <span>{resolution}</span>
        <span>{metadata?.fps_estimate == null ? "-" : `${metadata.fps_estimate.toFixed(1)} fps`}</span>
        <span>{age == null ? "no frame" : `${age.toFixed(age < 10 ? 1 : 0)}s old`}</span>
      </div>
    </section>
  );
}
