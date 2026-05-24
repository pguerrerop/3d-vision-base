export type RoiRect = {
  x: number;
  y: number;
  width: number;
  height: number;
};

export type RoiPolygon = Array<[number, number]>;

export function clampRoi(rect: RoiRect, imageWidth: number, imageHeight: number): RoiRect {
  const x0 = Math.max(0, Math.min(imageWidth - 1, Math.round(rect.x)));
  const y0 = Math.max(0, Math.min(imageHeight - 1, Math.round(rect.y)));
  const x1 = Math.max(x0 + 1, Math.min(imageWidth, Math.round(rect.x + rect.width)));
  const y1 = Math.max(y0 + 1, Math.min(imageHeight, Math.round(rect.y + rect.height)));
  return { x: x0, y: y0, width: x1 - x0, height: y1 - y0 };
}

export function pointsToRoi(a: { x: number; y: number }, b: { x: number; y: number }, imageWidth: number, imageHeight: number): RoiRect {
  const x = Math.min(a.x, b.x);
  const y = Math.min(a.y, b.y);
  const width = Math.abs(a.x - b.x);
  const height = Math.abs(a.y - b.y);
  return clampRoi({ x, y, width, height }, imageWidth, imageHeight);
}

export function mapClientToImagePoint(
  clientX: number,
  clientY: number,
  rect: { left: number; top: number; width: number; height: number },
  imageWidth: number,
  imageHeight: number
): { x: number; y: number } {
  const rx = rect.width > 0 ? (clientX - rect.left) / rect.width : 0;
  const ry = rect.height > 0 ? (clientY - rect.top) / rect.height : 0;
  const x = Math.max(0, Math.min(imageWidth, rx * imageWidth));
  const y = Math.max(0, Math.min(imageHeight, ry * imageHeight));
  return { x, y };
}

export function mapClientToImagePointStrict(
  clientX: number,
  clientY: number,
  rect: { left: number; top: number; width: number; height: number },
  imageWidth: number,
  imageHeight: number
): { x: number; y: number } | null {
  if (rect.width <= 0 || rect.height <= 0) return null;
  const rx = (clientX - rect.left) / rect.width;
  const ry = (clientY - rect.top) / rect.height;
  if (rx < 0 || rx > 1 || ry < 0 || ry > 1) return null;
  return {
    x: Math.max(0, Math.min(imageWidth, rx * imageWidth)),
    y: Math.max(0, Math.min(imageHeight, ry * imageHeight)),
  };
}

export function sourceToDisplay(
  point: { x: number; y: number },
  sourceWidth: number,
  sourceHeight: number,
  displayWidth: number,
  displayHeight: number
): { x: number; y: number } {
  const sx = sourceWidth > 0 ? point.x / sourceWidth : 0;
  const sy = sourceHeight > 0 ? point.y / sourceHeight : 0;
  return {
    x: sx * displayWidth,
    y: sy * displayHeight,
  };
}

export function clampPolygon(points: RoiPolygon, imageWidth: number, imageHeight: number): RoiPolygon {
  return points.map(([x, y]) => [
    Math.max(0, Math.min(imageWidth, Math.round(x))),
    Math.max(0, Math.min(imageHeight, Math.round(y))),
  ]);
}
