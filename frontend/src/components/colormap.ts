export type Rgb = readonly [number, number, number];

export type NamedColorMap = "turbo" | "viridis" | "magma" | "gray" | string;

// Exact OpenCV COLORMAP_TURBO LUT (256 RGB entries) extracted from
// cv2.applyColorMap. The rendered preview PNGs are produced by cv2 with this
// LUT after quantizing the normalized scalar to uint8. Reconstruction MUST use
// the same LUT (no polynomial approximation) for pixel-perfect parity with the
// rendered image.
const CV_TURBO_LUT: ReadonlyArray<readonly [number, number, number]> = [
  [48,18,59], [50,21,67], [51,24,74], [52,27,81], [53,30,88], [54,33,95], [55,36,102], [56,39,109],
  [57,42,115], [58,45,121], [59,47,128], [60,50,134], [61,53,139], [62,56,145], [63,59,151], [63,62,156],
  [64,64,162], [65,67,167], [65,70,172], [66,73,177], [66,75,181], [67,78,186], [68,81,191], [68,84,195],
  [68,86,199], [69,89,203], [69,92,207], [69,94,211], [70,97,214], [70,100,218], [70,102,221], [70,105,224],
  [70,107,227], [71,110,230], [71,113,233], [71,115,235], [71,118,238], [71,120,240], [71,123,242], [70,125,244],
  [70,128,246], [70,130,248], [70,133,250], [70,135,251], [69,138,252], [69,140,253], [68,143,254], [67,145,254],
  [66,148,255], [65,150,255], [64,153,255], [62,155,254], [61,158,254], [59,160,253], [58,163,252], [56,165,251],
  [55,168,250], [53,171,248], [51,173,247], [49,175,245], [47,178,244], [46,180,242], [44,183,240], [42,185,238],
  [40,188,235], [39,190,233], [37,192,231], [35,195,228], [34,197,226], [32,199,223], [31,201,221], [30,203,218],
  [28,205,216], [27,208,213], [26,210,210], [26,212,208], [25,213,205], [24,215,202], [24,217,200], [24,219,197],
  [24,221,194], [24,222,192], [24,224,189], [25,226,187], [25,227,185], [26,228,182], [28,230,180], [29,231,178],
  [31,233,175], [32,234,172], [34,235,170], [37,236,167], [39,238,164], [42,239,161], [44,240,158], [47,241,155],
  [50,242,152], [53,243,148], [56,244,145], [60,245,142], [63,246,138], [67,247,135], [70,248,132], [74,248,128],
  [78,249,125], [82,250,122], [85,250,118], [89,251,115], [93,252,111], [97,252,108], [101,253,105], [105,253,102],
  [109,254,98], [113,254,95], [117,254,92], [121,254,89], [125,255,86], [128,255,83], [132,255,81], [136,255,78],
  [139,255,75], [143,255,73], [146,255,71], [150,254,68], [153,254,66], [156,254,64], [159,253,63], [161,253,61],
  [164,252,60], [167,252,58], [169,251,57], [172,251,56], [175,250,55], [177,249,54], [180,248,54], [183,247,53],
  [185,246,53], [188,245,52], [190,244,52], [193,243,52], [195,241,52], [198,240,52], [200,239,52], [203,237,52],
  [205,236,52], [208,234,52], [210,233,53], [212,231,53], [215,229,53], [217,228,54], [219,226,54], [221,224,55],
  [223,223,55], [225,221,55], [227,219,56], [229,217,56], [231,215,57], [233,213,57], [235,211,57], [236,209,58],
  [238,207,58], [239,205,58], [241,203,58], [242,201,58], [244,199,58], [245,197,58], [246,195,58], [247,193,58],
  [248,190,57], [249,188,57], [250,186,57], [251,184,56], [251,182,55], [252,179,54], [252,177,54], [253,174,53],
  [253,172,52], [254,169,51], [254,167,50], [254,164,49], [254,161,48], [254,158,47], [254,155,45], [254,153,44],
  [254,150,43], [254,147,42], [254,144,41], [253,141,39], [253,138,38], [252,135,37], [252,132,35], [251,129,34],
  [251,126,33], [250,123,31], [249,120,30], [249,117,29], [248,114,28], [247,111,26], [246,108,25], [245,105,24],
  [244,102,23], [243,99,21], [242,96,20], [241,93,19], [240,91,18], [239,88,17], [237,85,16], [236,83,15],
  [235,80,14], [234,78,13], [232,75,12], [231,73,12], [229,71,11], [228,69,10], [226,67,10], [225,65,9],
  [223,63,8], [221,61,8], [220,59,7], [218,57,7], [216,55,6], [214,53,6], [212,51,5], [210,49,5],
  [208,47,5], [206,45,4], [204,43,4], [202,42,4], [200,40,3], [197,38,3], [195,37,3], [193,35,2],
  [190,33,2], [188,32,2], [185,30,2], [183,29,2], [180,27,1], [178,26,1], [175,24,1], [172,23,1],
  [169,22,1], [167,20,1], [164,19,1], [161,18,1], [158,16,1], [155,15,1], [152,14,1], [149,13,1],
  [146,11,1], [142,10,1], [139,9,2], [136,8,2], [133,7,2], [129,6,2], [126,5,2], [122,4,3],
];

export function turboColor01(t: number): [number, number, number] {
  const x = Math.min(1, Math.max(0, t));
  // Match cv2's quantization: t -> uint8 index, then LUT lookup.
  const idx = Math.max(0, Math.min(255, Math.round(x * 255)));
  const entry = CV_TURBO_LUT[idx];
  return [entry[0], entry[1], entry[2]];
}

// Viridis stops (sampled from matplotlib's viridis cmap; t in [0,1]).
const VIRIDIS_STOPS: Array<[number, [number, number, number]]> = [
  [0.0, [68, 1, 84]],
  [0.25, [59, 82, 139]],
  [0.5, [33, 145, 140]],
  [0.75, [94, 201, 98]],
  [1.0, [253, 231, 37]],
];

// Magma stops (sampled from matplotlib's magma cmap; t in [0,1]).
const MAGMA_STOPS: Array<[number, [number, number, number]]> = [
  [0.0, [0, 0, 4]],
  [0.25, [80, 18, 123]],
  [0.5, [183, 55, 121]],
  [0.75, [251, 136, 97]],
  [1.0, [252, 253, 191]],
];

const GRAY_STOPS: Array<[number, [number, number, number]]> = [
  [0.0, [0, 0, 0]],
  [1.0, [255, 255, 255]],
];

function sampleStops(stops: Array<[number, [number, number, number]]>, t: number): [number, number, number] {
  const x = Math.min(1, Math.max(0, t));
  for (let i = 1; i < stops.length; i += 1) {
    const [tPrev, cPrev] = stops[i - 1];
    const [tNext, cNext] = stops[i];
    if (x <= tNext) {
      const span = Math.max(1e-9, tNext - tPrev);
      const local = (x - tPrev) / span;
      return [
        clamp255(cPrev[0] + (cNext[0] - cPrev[0]) * local),
        clamp255(cPrev[1] + (cNext[1] - cPrev[1]) * local),
        clamp255(cPrev[2] + (cNext[2] - cPrev[2]) * local),
      ];
    }
  }
  const [, last] = stops[stops.length - 1];
  return [last[0], last[1], last[2]];
}

export function sampleColorMap(name: NamedColorMap, t: number): [number, number, number] {
  switch (String(name || "").toLowerCase()) {
    case "viridis":
      return sampleStops(VIRIDIS_STOPS, t);
    case "magma":
      return sampleStops(MAGMA_STOPS, t);
    case "gray":
    case "grey":
    case "grayscale":
      return sampleStops(GRAY_STOPS, t);
    case "turbo":
    default:
      return turboColor01(t);
  }
}

// Build a CSS linear-gradient string from a named colormap so the legend
// renders the same LUT the image renderer uses (no hard-coded gradients).
// `direction` controls vertical orientation (top-to-bottom value).
export function cssGradientForColorMap(name: NamedColorMap, samples = 16, direction: "higher_is_hotter" | "lower_is_hotter" = "higher_is_hotter"): string {
  const stops: string[] = [];
  for (let i = 0; i <= samples; i += 1) {
    const t = i / samples;
    // Gradient is written top->bottom (180deg). For higher_is_hotter, the top
    // should be hot (t=1) and the bottom cool (t=0), so we invert the sample.
    const tSample = direction === "higher_is_hotter" ? (1 - t) : t;
    const [r, g, b] = sampleColorMap(name, tSample);
    const pct = Math.round(t * 100);
    stops.push(`rgb(${r}, ${g}, ${b}) ${pct}%`);
  }
  return `linear-gradient(180deg, ${stops.join(", ")})`;
}

export function scalarToRgb(name: NamedColorMap, value: number | null, min: number, max: number, valid: boolean): [number, number, number] {
  if (!valid || value == null || !Number.isFinite(value)) return [0, 0, 0];
  const range = Math.max(1e-9, max - min);
  const clipped = Math.min(max, Math.max(min, value));
  return sampleColorMap(name, (clipped - min) / range);
}

// Back-compat shim: prefer scalarToRgb(name, ...) for new callers.
export function scalarToTurboRgb(value: number | null, min: number, max: number, valid: boolean): [number, number, number] {
  return scalarToRgb("turbo", value, min, max, valid);
}

export function inferTurboTFromRgb(rgb: Rgb): number {
  return inferColorMapTFromRgb("turbo", rgb);
}

export function inferColorMapTFromRgb(name: NamedColorMap, rgb: Rgb): number {
  // Dense lookup to estimate normalized scalar that produced an RGB pixel.
  // This is used for scalar-domain diffing against rendered output.
  let bestT = 0;
  let bestDist = Number.POSITIVE_INFINITY;
  for (let i = 0; i <= 4096; i += 1) {
    const t = i / 4096;
    const [tr, tg, tb] = sampleColorMap(name, t);
    const dr = tr - rgb[0];
    const dg = tg - rgb[1];
    const db = tb - rgb[2];
    const d = (dr * dr) + (dg * dg) + (db * db);
    if (d < bestDist) {
      bestDist = d;
      bestT = t;
    }
  }
  return bestT;
}

function clamp255(n: number): number {
  return Math.max(0, Math.min(255, Math.round(n)));
}
