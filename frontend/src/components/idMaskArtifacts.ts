import type { StudioArtifact } from "./studioWorkspaceModel";

// Backend convention (see stages_25d.py, e.g. `(fragment_labels % 255).astype(np.uint8)`): "*_id_mask"
// artifacts store small integer component/fragment ids as raw grayscale pixel values. Displayed
// directly, low-valued ids (1, 2, 3...) all land near black and are visually indistinguishable --
// the exact issue the backend's own `_label_colormap_overlay` docstring calls out for its "_overlay"
// siblings. This module recolors the same way, client-side, for the plain id-mask artifacts that
// don't get that treatment.
const ID_MASK_PATTERN = /(^|_)id_mask$/i;

export function isIdMaskArtifact(artifact: StudioArtifact): boolean {
  if (artifact.kind !== "image") return false;
  return ID_MASK_PATTERN.test(artifact.artifact_id);
}

const GOLDEN_ANGLE_DEG = 137.508;

function hslToRgb(hueDeg: number, saturation: number, lightness: number): [number, number, number] {
  const c = (1 - Math.abs(2 * lightness - 1)) * saturation;
  const hp = hueDeg / 60;
  const x = c * (1 - Math.abs((hp % 2) - 1));
  let seg: [number, number, number] = [0, 0, 0];
  if (hp < 1) seg = [c, x, 0];
  else if (hp < 2) seg = [x, c, 0];
  else if (hp < 3) seg = [0, c, x];
  else if (hp < 4) seg = [0, x, c];
  else if (hp < 5) seg = [x, 0, c];
  else seg = [c, 0, x];
  const m = lightness - c / 2;
  return [Math.round((seg[0] + m) * 255), Math.round((seg[1] + m) * 255), Math.round((seg[2] + m) * 255)];
}

// Golden-angle hue stepping gives maximally-separated hues for an arbitrary, unknown-in-advance
// number of ids without needing to port the backend's 256-entry turbo colormap LUT into the browser.
export function categoricalLabelColor(index: number): [number, number, number] {
  const hue = (index * GOLDEN_ANGLE_DEG) % 360;
  return hslToRgb(hue, 0.65, 0.52);
}

// Mutates `imageData` in place: every unique non-zero grayscale value (an integer label id) gets
// remapped to a distinct categorical color. Zero (no component) stays black.
export function colorizeIdMaskImageData(imageData: ImageData): void {
  const { data } = imageData;
  const uniqueValues = new Set<number>();
  for (let i = 0; i < data.length; i += 4) {
    if (data[i] !== 0) uniqueValues.add(data[i]);
  }
  if (uniqueValues.size === 0) return;
  const colorByValue = new Map<number, [number, number, number]>();
  Array.from(uniqueValues)
    .sort((a, b) => a - b)
    .forEach((value, index) => colorByValue.set(value, categoricalLabelColor(index)));
  for (let i = 0; i < data.length; i += 4) {
    const color = colorByValue.get(data[i]);
    if (!color) continue;
    data[i] = color[0];
    data[i + 1] = color[1];
    data[i + 2] = color[2];
  }
}
