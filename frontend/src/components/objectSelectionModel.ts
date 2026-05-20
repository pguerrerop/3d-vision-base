export function objectDisplayId(objectId: number | string): string {
  const numeric = typeof objectId === "string" ? Number(objectId) : objectId;
  if (!Number.isFinite(numeric) || numeric <= 0) return "object_000";
  return `object_${Math.trunc(numeric).toString().padStart(3, "0")}`;
}

export function objectIdFromOverlay(overlay: { object_id?: number | null }): number | null {
  return typeof overlay.object_id === "number" ? overlay.object_id : null;
}

export function objectIdFromMeasurementRow(row: { object_id?: number | null }): number | null {
  return typeof row.object_id === "number" ? row.object_id : null;
}

export function containsRawUuidLabel(label: string): boolean {
  return /[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}/i.test(label);
}
