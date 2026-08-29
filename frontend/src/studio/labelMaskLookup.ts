export type LabelEntry = { id: number; isRejected: boolean };

// The backend writes per-pixel id masks as 8-bit grayscale PNGs (`label % 255` per pixel), so a
// candidate's own known id/label is looked up by the same modulus -- no color sampling or
// tolerance matching needed, and (unlike sampling at a candidate's centroid) no dependency on the
// centroid actually lying within its own possibly non-convex/ring-shaped region.
export function buildLabelLookup(rows: Array<{ id: string | number }>, rejectedIds: Set<number>): Map<number, LabelEntry> {
  const lookup = new Map<number, LabelEntry>();
  for (const row of rows) {
    const id = Number(row.id);
    if (!Number.isFinite(id) || id <= 0) continue;
    lookup.set(id % 255, { id, isRejected: rejectedIds.has(id) });
  }
  return lookup;
}
