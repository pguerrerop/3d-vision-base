import { useEffect, useState } from "react";
import { finiteOrDefault, resolveBinaryMaskReference, type MaskReferenceResolution } from "../../binaryMaskArtifacts";
import type { MaskRenderedAssets } from "./maskRenderingTypes";

type ImageLoadState = {
  maskSize: { width: number; height: number } | null;
  sourceSize: { width: number; height: number } | null;
  sourceFailed: boolean;
};

const MASK_ON = [34, 197, 94] as const;
const MASK_OFF = [198, 208, 218] as const;
const MASK_INVALID = [102, 116, 139] as const;

export function useMaskRenderAssets(args: {
  maskUrl: string | null;
  sourceUrl: string | null;
  invalidUrl: string | null;
}) {
  const { maskUrl, sourceUrl, invalidUrl } = args;
  const [rendered, setRendered] = useState<MaskRenderedAssets | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loadState, setLoadState] = useState<ImageLoadState>({
    maskSize: null,
    sourceSize: null,
    sourceFailed: false,
  });

  useEffect(() => {
    let cancelled = false;
    let nextUrls: string[] = [];
    async function render() {
      if (!maskUrl) return;
      try {
        const [maskBitmap, sourceBitmap, invalidBitmap] = await Promise.all([
          loadBitmap(maskUrl),
          sourceUrl ? loadBitmap(sourceUrl).catch(() => null) : Promise.resolve(null),
          invalidUrl ? loadBitmap(invalidUrl).catch(() => null) : Promise.resolve(null),
        ]);
        if (cancelled) return;
        const nextLoadState: ImageLoadState = {
          maskSize: { width: finiteOrDefault(maskBitmap.width, 1), height: finiteOrDefault(maskBitmap.height, 1) },
          sourceSize: sourceBitmap ? { width: finiteOrDefault(sourceBitmap.width, 1), height: finiteOrDefault(sourceBitmap.height, 1) } : null,
          sourceFailed: Boolean(sourceUrl && !sourceBitmap),
        };
        setLoadState((current) => sameLoadState(current, nextLoadState) ? current : nextLoadState);
        const assets = await buildRenderAssets(maskBitmap, sourceBitmap, invalidBitmap);
        nextUrls = [assets.maskOnlyUrl, assets.overlayUrl, assets.boundaryUrl];
        if (cancelled) {
          nextUrls.forEach(URL.revokeObjectURL);
          return;
        }
        setError(null);
        setRendered((current) => {
          if (current) {
            URL.revokeObjectURL(current.maskOnlyUrl);
            URL.revokeObjectURL(current.overlayUrl);
            URL.revokeObjectURL(current.boundaryUrl);
          }
          return assets;
        });
      } catch (nextError) {
        if (cancelled) return;
        setError(nextError instanceof Error ? nextError.message : "Failed to render binary mask preview.");
      }
    }
    setLoadState((current) => (current.maskSize || current.sourceSize || current.sourceFailed
      ? { maskSize: null, sourceSize: null, sourceFailed: false }
      : current));
    void render();
    return () => {
      cancelled = true;
      nextUrls.forEach(URL.revokeObjectURL);
    };
  }, [invalidUrl, maskUrl, sourceUrl]);

  useEffect(() => {
    return () => {
      if (!rendered) return;
      URL.revokeObjectURL(rendered.maskOnlyUrl);
      URL.revokeObjectURL(rendered.overlayUrl);
      URL.revokeObjectURL(rendered.boundaryUrl);
    };
  }, [rendered]);

  return { rendered, error, loadState };
}

export function validateMaskResolutionForLoadedImages(
  resolution: ReturnType<typeof resolveBinaryMaskReference>,
  loadState: ImageLoadState,
): MaskReferenceResolution {
  if (!resolution.sourceArtifact) return resolution;
  if (loadState.sourceFailed) {
    return {
      ...resolution,
      alignment: "unavailable" as const,
      reason: "reference artifact found but image URL is unavailable",
      warning: "Reference artifact found but image URL is unavailable.",
      sourceUrl: undefined,
    };
  }
  if (!loadState.maskSize || !loadState.sourceSize) return resolution;
  if (
    loadState.maskSize.width !== loadState.sourceSize.width
    || loadState.maskSize.height !== loadState.sourceSize.height
  ) {
    return {
      ...resolution,
      alignment: "dimension_mismatch" as const,
      reason: "reference dimensions do not match mask",
      warning: "Reference dimensions do not match mask. Overlay disabled.",
    };
  }
  return resolution;
}

function sameLoadState(a: ImageLoadState, b: ImageLoadState): boolean {
  return (
    a.sourceFailed === b.sourceFailed
    && sameSize(a.maskSize, b.maskSize)
    && sameSize(a.sourceSize, b.sourceSize)
  );
}

function sameSize(
  a: { width: number; height: number } | null,
  b: { width: number; height: number } | null,
): boolean {
  if (!a && !b) return true;
  if (!a || !b) return false;
  return a.width === b.width && a.height === b.height;
}

async function buildRenderAssets(
  maskBitmap: ImageBitmap,
  sourceBitmap: ImageBitmap | null,
  invalidBitmap: ImageBitmap | null,
): Promise<MaskRenderedAssets> {
  const width = Math.max(1, finiteOrDefault(maskBitmap.width, 1));
  const height = Math.max(1, finiteOrDefault(maskBitmap.height, 1));
  const maskPixels = readPixels(maskBitmap, width, height);
  const invalidPixels = invalidBitmap ? readPixels(invalidBitmap, width, height) : null;
  const sourcePixels = sourceBitmap ? readPixels(sourceBitmap, width, height) : null;

  const maskOnly = new Uint8ClampedArray(width * height * 4);
  const overlay = new Uint8ClampedArray(width * height * 4);
  const boundary = new Uint8ClampedArray(width * height * 4);
  let hasInvalidData = false;

  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const idx = (y * width) + x;
      const pixel = idx * 4;
      const on = pixelIntensity(maskPixels, pixel) >= 127;
      const valid = invalidPixels ? pixelIntensity(invalidPixels, pixel) >= 127 : true;
      const isBoundary = on && touchesOffNeighbor(maskPixels, x, y, width, height);
      if (!valid) hasInvalidData = true;

      const checker = checkerShade(x, y);
      const [baseR, baseG, baseB] = valid ? checker : MASK_INVALID;
      const [maskR, maskG, maskB] = valid ? (on ? MASK_ON : MASK_OFF) : MASK_INVALID;
      maskOnly[pixel + 0] = blendChannel(baseR, maskR, valid ? (on ? 0.94 : 0.88) : 1);
      maskOnly[pixel + 1] = blendChannel(baseG, maskG, valid ? (on ? 0.94 : 0.88) : 1);
      maskOnly[pixel + 2] = blendChannel(baseB, maskB, valid ? (on ? 0.94 : 0.88) : 1);
      maskOnly[pixel + 3] = 255;

      const sourceR = sourcePixels ? sourcePixels[pixel + 0] : checker[0];
      const sourceG = sourcePixels ? sourcePixels[pixel + 1] : checker[1];
      const sourceB = sourcePixels ? sourcePixels[pixel + 2] : checker[2];
      overlay[pixel + 0] = valid ? (on ? MASK_ON[0] : sourceR) : MASK_INVALID[0];
      overlay[pixel + 1] = valid ? (on ? MASK_ON[1] : sourceG) : MASK_INVALID[1];
      overlay[pixel + 2] = valid ? (on ? MASK_ON[2] : sourceB) : MASK_INVALID[2];
      overlay[pixel + 3] = valid ? (on ? 255 : 0) : 160;

      boundary[pixel + 0] = isBoundary ? MASK_ON[0] : 0;
      boundary[pixel + 1] = isBoundary ? MASK_ON[1] : 0;
      boundary[pixel + 2] = isBoundary ? MASK_ON[2] : 0;
      boundary[pixel + 3] = isBoundary ? 255 : 0;
    }
  }

  return {
    maskOnlyUrl: await pixelsToObjectUrl(maskOnly, width, height),
    overlayUrl: await pixelsToObjectUrl(overlay, width, height),
    boundaryUrl: await pixelsToObjectUrl(boundary, width, height),
    hasInvalidData,
  };
}

function readPixels(bitmap: ImageBitmap, width: number, height: number): Uint8ClampedArray {
  const safeWidth = Math.max(1, finiteOrDefault(width, 1));
  const safeHeight = Math.max(1, finiteOrDefault(height, 1));
  const canvas = document.createElement("canvas");
  canvas.width = safeWidth;
  canvas.height = safeHeight;
  const ctx = canvas.getContext("2d");
  if (!ctx) throw new Error("Canvas 2D context unavailable.");
  ctx.clearRect(0, 0, safeWidth, safeHeight);
  ctx.drawImage(bitmap, 0, 0, safeWidth, safeHeight);
  return ctx.getImageData(0, 0, safeWidth, safeHeight).data;
}

function pixelIntensity(pixels: Uint8ClampedArray, pixel: number): number {
  return Math.round((pixels[pixel + 0] + pixels[pixel + 1] + pixels[pixel + 2]) / 3);
}

function checkerShade(x: number, y: number): [number, number, number] {
  const light = ((Math.floor(x / 12) + Math.floor(y / 12)) % 2) === 0;
  return light ? [236, 241, 245] : [220, 228, 236];
}

function touchesOffNeighbor(maskPixels: Uint8ClampedArray, x: number, y: number, width: number, height: number): boolean {
  const neighbors = [
    [0, -1],
    [1, 0],
    [0, 1],
    [-1, 0],
  ];
  for (const [dx, dy] of neighbors) {
    const nx = x + dx;
    const ny = y + dy;
    if (nx < 0 || ny < 0 || nx >= width || ny >= height) return true;
    const neighborPixel = ((ny * width) + nx) * 4;
    if (pixelIntensity(maskPixels, neighborPixel) < 127) return true;
  }
  return false;
}

function blendChannel(base: number, accent: number, alpha: number): number {
  const safeAlpha = Math.max(0, Math.min(1, finiteOrDefault(alpha, 0)));
  return Math.round((base * (1 - safeAlpha)) + (accent * safeAlpha));
}

async function pixelsToObjectUrl(pixels: Uint8ClampedArray, width: number, height: number): Promise<string> {
  const safeWidth = Math.max(1, finiteOrDefault(width, 1));
  const safeHeight = Math.max(1, finiteOrDefault(height, 1));
  const canvas = document.createElement("canvas");
  canvas.width = safeWidth;
  canvas.height = safeHeight;
  const ctx = canvas.getContext("2d");
  if (!ctx) throw new Error("Canvas 2D context unavailable.");
  ctx.clearRect(0, 0, safeWidth, safeHeight);
  ctx.putImageData(new ImageData(new Uint8ClampedArray(pixels), safeWidth, safeHeight), 0, 0);
  const blob = await new Promise<Blob | null>((resolve) => canvas.toBlob(resolve, "image/png"));
  if (!blob) throw new Error("Unable to encode binary mask preview.");
  return URL.createObjectURL(blob);
}

async function loadBitmap(url: string): Promise<ImageBitmap> {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Unable to load ${url}`);
  return createImageBitmap(await response.blob());
}
