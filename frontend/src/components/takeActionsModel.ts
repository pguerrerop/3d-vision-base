export function canConfirmPermanentDelete(takeId: string, typed: string): boolean {
  return typed.trim() === takeId;
}
