import type { RuntimePreviewMetadata, RuntimeState } from "../api/client";

export type LivePreviewStatus = "connected" | "stale" | "disconnected" | "unavailable";

export function livePreviewStatus(state: RuntimeState | null, metadata: RuntimePreviewMetadata | null): LivePreviewStatus {
  if (!state?.acquisition_connected) {
    return metadata?.timestamp ? "disconnected" : "unavailable";
  }
  if (!metadata?.timestamp || metadata.error === "no preview available") {
    return "unavailable";
  }
  if (metadata.stale || state.preview_stale) {
    return "stale";
  }
  return "connected";
}

export function livePreviewLabel(status: LivePreviewStatus): string {
  if (status === "connected") return "CONNECTED";
  if (status === "stale") return "STALE";
  if (status === "disconnected") return "DISCONNECTED";
  return "NO PREVIEW AVAILABLE";
}

export function imageAgeSeconds(timestamp: string | null | undefined, now = Date.now()): number | null {
  if (!timestamp) return null;
  const parsed = Date.parse(timestamp);
  if (Number.isNaN(parsed)) return null;
  return Math.max(0, (now - parsed) / 1000);
}
