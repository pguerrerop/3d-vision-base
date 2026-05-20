export type HistogramStats = {
  bins: number[];
  min: number;
  max: number;
  mean: number;
  sampledPixels: number;
  width: number;
  height: number;
};

export type HistogramLoadResult =
  | { ok: true; stats: HistogramStats }
  | { ok: false; reason: string };

type ImageLike = {
  width: number;
  height: number;
  onload: ((...args: any[]) => void) | null;
  onerror: ((...args: any[]) => void) | null;
  crossOrigin?: string | null;
  src: string;
};

type CanvasContextLike = {
  drawImage: (image: ImageLike, sx: number, sy: number, sw?: number, sh?: number) => void;
  getImageData: (sx: number, sy: number, sw: number, sh: number) => { data: Uint8ClampedArray };
};

type CanvasLike = {
  width: number;
  height: number;
  getContext: (kind: "2d") => CanvasContextLike | null;
};

export type HistogramDeps = {
  createImage: () => ImageLike;
  createCanvas: () => CanvasLike;
  setTimer: (fn: () => void, ms: number) => unknown;
  clearTimer: (id: unknown) => void;
};

const DEFAULT_MAX_DIM = 512;

export function boundedDimensions(width: number, height: number, maxDim = DEFAULT_MAX_DIM): { width: number; height: number } {
  if (width <= 0 || height <= 0) return { width: 1, height: 1 };
  const scale = Math.min(1, maxDim / Math.max(width, height));
  return { width: Math.max(1, Math.round(width * scale)), height: Math.max(1, Math.round(height * scale)) };
}

export function computeLuminanceHistogram(
  data: Uint8ClampedArray,
  width: number,
  height: number,
  bins = 32
): HistogramStats {
  const counts = new Array(bins).fill(0) as number[];
  let min = 255;
  let max = 0;
  let sum = 0;
  let pixels = 0;
  for (let i = 0; i + 2 < data.length; i += 4) {
    const lum = Math.round(0.2126 * data[i] + 0.7152 * data[i + 1] + 0.0722 * data[i + 2]);
    const idx = Math.min(bins - 1, Math.floor((lum / 256) * bins));
    counts[idx] += 1;
    if (lum < min) min = lum;
    if (lum > max) max = lum;
    sum += lum;
    pixels += 1;
  }
  return {
    bins: counts,
    min: pixels ? min : 0,
    max: pixels ? max : 0,
    mean: pixels ? sum / pixels : 0,
    sampledPixels: pixels,
    width,
    height,
  };
}

export async function loadHistogramFromImageUrl(
  url: string,
  options?: { maxDim?: number; timeoutMs?: number; bins?: number; deps?: HistogramDeps }
): Promise<HistogramLoadResult> {
  const maxDim = options?.maxDim ?? DEFAULT_MAX_DIM;
  const timeoutMs = options?.timeoutMs ?? 5000;
  const bins = options?.bins ?? 32;
  const deps: HistogramDeps = options?.deps ?? {
    createImage: () => new Image(),
    createCanvas: () => document.createElement("canvas") as unknown as CanvasLike,
    setTimer: (fn, ms) => window.setTimeout(fn, ms),
    clearTimer: (id) => window.clearTimeout(id as number),
  };

  return new Promise<HistogramLoadResult>((resolve) => {
    let settled = false;
    const settle = (value: HistogramLoadResult) => {
      if (settled) return;
      settled = true;
      deps.clearTimer(timeoutId);
      resolve(value);
    };

    const timeoutId = deps.setTimer(() => {
      settle({ ok: false, reason: "Histogram computation failed: timeout" });
    }, timeoutMs);

    const image = deps.createImage();
    image.crossOrigin = "anonymous";
    image.onload = () => {
      try {
        const bounded = boundedDimensions(image.width, image.height, maxDim);
        const canvas = deps.createCanvas();
        canvas.width = bounded.width;
        canvas.height = bounded.height;
        const ctx = canvas.getContext("2d");
        if (!ctx) {
          settle({ ok: false, reason: "Histogram unavailable for this source image." });
          return;
        }
        try {
          ctx.drawImage(image, 0, 0, bounded.width, bounded.height);
          const data = ctx.getImageData(0, 0, bounded.width, bounded.height);
          const stats = computeLuminanceHistogram(data.data, bounded.width, bounded.height, bins);
          settle({ ok: true, stats });
        } catch {
          settle({ ok: false, reason: "Histogram unavailable for this source image." });
        }
      } catch {
        settle({ ok: false, reason: "Histogram computation failed: processing error" });
      }
    };

    image.onerror = () => {
      settle({ ok: false, reason: "Histogram unavailable for this source image." });
    };

    image.src = url;
  });
}
