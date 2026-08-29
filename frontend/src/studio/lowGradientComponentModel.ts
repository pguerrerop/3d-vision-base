import { useEffect, useState } from "react";
import type { BlobCandidate } from "./blobDetectionModel.js";

export type LowGradientComponents = {
  components: BlobCandidate[];
  rejected: BlobCandidate[];
};

function num(value: unknown): number | undefined {
  const n = Number(value);
  return Number.isFinite(n) ? n : undefined;
}

function pair(value: unknown): [number, number] | undefined {
  if (!Array.isArray(value) || value.length < 2) return undefined;
  const x = num(value[0]);
  const y = num(value[1]);
  return x == null || y == null ? undefined : [x, y];
}

function quad(value: unknown): [number, number, number, number] | undefined {
  if (!Array.isArray(value) || value.length < 4) return undefined;
  const a = num(value[0]); const b = num(value[1]); const c = num(value[2]); const d = num(value[3]);
  return a == null || b == null || c == null || d == null ? undefined : [a, b, c, d];
}

function rowsFromPayload(payload: unknown): Record<string, unknown>[] {
  const array = Array.isArray(payload)
    ? payload
    : (payload && typeof payload === "object" && Array.isArray((payload as Record<string, unknown>).entries))
      ? (payload as Record<string, unknown>).entries as unknown[]
      : [];
  return array.filter((item): item is Record<string, unknown> => typeof item === "object" && item !== null);
}

function normalizeRow(row: Record<string, unknown>): BlobCandidate {
  const idValue = row.blob_id ?? row.component_id ?? row.id;
  return {
    id: (typeof idValue === "string" && idValue.trim()) ? idValue : (num(idValue) ?? 0),
    areaPx: num(row.area_px),
    centroid: pair(row.centroid),
    bbox: quad(row.bbox),
    aspectRatio: num(row.aspect_ratio),
    solidity: num(row.solidity),
    touchesBorder: Boolean(row.touches_frame_border ?? row.touches_roi_border),
    role: typeof row.role === "string" ? row.role : undefined,
    meanHeightMm: num(row.mean_z ?? row.median_z),
    rejectionReason: typeof row.rejected_reason === "string" ? row.rejected_reason : undefined,
    status: row.rejected ? "rejected" : "candidate",
  };
}

export function normalizeLowGradientComponents(rawRows: unknown): LowGradientComponents {
  const rows = rowsFromPayload(rawRows).map(normalizeRow);
  return {
    components: rows.filter((row) => row.status !== "rejected"),
    rejected: rows.filter((row) => row.status === "rejected"),
  };
}

export function useLowGradientComponents(
  url: string | null | undefined,
): { loading: boolean; error: boolean } & LowGradientComponents {
  const [state, setState] = useState<{ loading: boolean; error: boolean } & LowGradientComponents>({
    loading: true,
    error: false,
    components: [],
    rejected: [],
  });

  useEffect(() => {
    let cancelled = false;
    if (!url) {
      setState({ loading: false, error: false, components: [], rejected: [] });
      return;
    }
    setState((prev) => ({ ...prev, loading: true, error: false }));
    void (async () => {
      try {
        const response = await fetch(url);
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const payload = await response.json();
        if (cancelled) return;
        setState({ loading: false, error: false, ...normalizeLowGradientComponents(payload) });
      } catch {
        if (!cancelled) setState({ loading: false, error: true, components: [], rejected: [] });
      }
    })();
    return () => { cancelled = true; };
  }, [url]);

  return state;
}
