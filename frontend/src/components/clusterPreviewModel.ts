import type { StudioArtifact } from "./studioWorkspaceModel";
import type { ClusterRow } from "./clusterDecisionModel";

export function clusterColor(clusterId: string): string {
  let hash = 0;
  for (let i = 0; i < clusterId.length; i += 1) {
    hash = (hash * 31 + clusterId.charCodeAt(i)) >>> 0;
  }
  const hue = hash % 360;
  return `hsl(${hue}, 70%, 45%)`;
}

export function parseMemberId(value: string | number): number | null {
  if (typeof value === "number" && Number.isFinite(value)) return value;
  const raw = String(value ?? "").trim();
  if (!raw) return null;
  const direct = Number(raw);
  if (Number.isFinite(direct)) return direct;
  const match = raw.match(/(\d+)\s*$/);
  if (!match) return null;
  const parsed = Number(match[1]);
  return Number.isFinite(parsed) ? parsed : null;
}

export function resolveBlobIdMaskArtifact(artifacts: StudioArtifact[]): StudioArtifact | null {
  const find = (artifactId: string) => artifacts.find((artifact) => artifact.artifact_id === artifactId && artifact.path) ?? null;
  return find("height_split_blob_id_mask")
    ?? find("height_aware_blob_id_mask")
    ?? find("low_gradient_blob_id_mask");
}

export function memberIdsForClusterPreview(row: ClusterRow, maskArtifact: StudioArtifact | null): number[] {
  const maskId = String(maskArtifact?.artifact_id ?? "");
  const preferFragments = maskId.includes("split") || row.fragmentIds.length > 0;
  const source = preferFragments ? row.fragmentIds : row.componentIds;
  return source.map(parseMemberId).filter((value): value is number => value != null);
}

export function hslToRgb(hsl: string): [number, number, number] {
  const match = hsl.match(/hsl\((\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?)%,\s*(\d+(?:\.\d+)?)%\)/i);
  if (!match) return [59, 130, 246];
  const h = Number(match[1]) / 360;
  const s = Number(match[2]) / 100;
  const l = Number(match[3]) / 100;
  if (s === 0) {
    const gray = Math.round(l * 255);
    return [gray, gray, gray];
  }
  const hue2rgb = (p: number, q: number, t: number) => {
    let value = t;
    if (value < 0) value += 1;
    if (value > 1) value -= 1;
    if (value < 1 / 6) return p + (q - p) * 6 * value;
    if (value < 1 / 2) return q;
    if (value < 2 / 3) return p + (q - p) * (2 / 3 - value) * 6;
    return p;
  };
  const q = l < 0.5 ? l * (1 + s) : l + s - l * s;
  const p = 2 * l - q;
  return [
    Math.round(hue2rgb(p, q, h + 1 / 3) * 255),
    Math.round(hue2rgb(p, q, h) * 255),
    Math.round(hue2rgb(p, q, h - 1 / 3) * 255),
  ];
}

export async function loadImageElement(url: string): Promise<HTMLImageElement> {
  return await new Promise((resolve, reject) => {
    const image = new Image();
    image.crossOrigin = "anonymous";
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error(`Failed to load image: ${url}`));
    image.src = url;
  });
}
