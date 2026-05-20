export function captureDisabledReason(datasetId: string, sessionId: string): string | null {
  if (!datasetId) return "Select a dataset first";
  if (!sessionId) return "Select an experiment session first";
  return null;
}

export function buildCapturePayload(input: {
  datasetId: string;
  sessionId: string;
  friendlyName?: string;
  tagsText?: string;
  expectedClass?: string;
  expectedDiameterMm?: string;
  notes?: string;
}) {
  return {
    dataset_id: input.datasetId,
    dataset_session_id: input.sessionId,
    friendly_name: input.friendlyName?.trim() || null,
    tags: (input.tagsText || "").split(",").map((item) => item.trim()).filter(Boolean),
    expected_class: input.expectedClass?.trim() || null,
    expected_diameter_mm: input.expectedDiameterMm?.trim() ? Number(input.expectedDiameterMm) : null,
    notes: input.notes?.trim() || null,
  };
}

export function nextSelectedTakeAfterCapture(currentTakeId: string, capturedTakeId: string): string {
  return capturedTakeId || currentTakeId;
}
