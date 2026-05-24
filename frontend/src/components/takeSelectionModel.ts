export type ResolveNextSelectedTakeIdArgs = {
  currentSelectedTakeId: string | null | undefined;
  availableTakeIds: string[];
  preferredTakeId?: string | null | undefined;
};

export function resolveNextSelectedTakeId({
  currentSelectedTakeId,
  availableTakeIds,
  preferredTakeId,
}: ResolveNextSelectedTakeIdArgs): string | null {
  if (!availableTakeIds.length) return null;
  if (currentSelectedTakeId && availableTakeIds.includes(currentSelectedTakeId)) {
    return currentSelectedTakeId;
  }
  if (preferredTakeId && availableTakeIds.includes(preferredTakeId)) {
    return preferredTakeId;
  }
  return availableTakeIds[0] ?? null;
}
