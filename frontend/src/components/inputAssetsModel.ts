import type { CaptureModality, TakeDetail } from "../api/client";

export type InputAssetTab = CaptureModality | "metadata";

export function availableInputTabs(detail: TakeDetail | null): InputAssetTab[] {
  const modalities = detail?.modalities?.length ? detail.modalities : fallbackModalities(detail?.metadata);
  return [...modalities, "metadata"];
}

function fallbackModalities(metadata: Record<string, unknown> | null | undefined): CaptureModality[] {
  const files = typeof metadata?.files === "object" && metadata.files ? metadata.files as Record<string, unknown> : {};
  const modalities: CaptureModality[] = [];
  if (files.point_cloud || files.point_cloud_npz) modalities.push("point_cloud");
  if (files.heightmap || files.height) modalities.push("heightmap");
  if (files.reflectance) modalities.push("reflectance");
  if (files.rgb) modalities.push("rgb");
  if (files.rgb_video) modalities.push("rgb_video");
  if (files.laser_rgb || files.laser_overlay || files.laser_image) modalities.push("laser_rgb");
  return modalities.length ? modalities : ["point_cloud"];
}
