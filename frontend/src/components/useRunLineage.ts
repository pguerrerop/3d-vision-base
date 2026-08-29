import { useCallback, useEffect, useState } from "react";

import { api, type PipelineRunLineage } from "../api/client";

export function useRunLineage(pipelineId: string | null, runId: string | null) {
  const [lineage, setLineage] = useState<PipelineRunLineage | null>(null);
  const [loading, setLoading] = useState(false);
  const [generating, setGenerating] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [refreshToken, setRefreshToken] = useState(0);

  useEffect(() => {
    let cancelled = false;
    if (!pipelineId || !runId) {
      setLineage(null);
      return () => { cancelled = true; };
    }
    setLoading(true);
    void api
      .pipelineRunLineage(pipelineId, runId)
      .then((response) => {
        if (cancelled) return;
        setLineage(response);
      })
      .catch(() => {
        if (cancelled) return;
        setLineage(null);
      })
      .finally(() => {
        if (cancelled) return;
        setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [pipelineId, runId, refreshToken]);

  const refresh = useCallback(() => setRefreshToken((token) => token + 1), []);

  const generateComparison = useCallback(async () => {
    if (!pipelineId || !runId) return null;
    setGenerating(true);
    setError(null);
    try {
      const comparison = await api.generateRunComparisonToParent(pipelineId, runId);
      refresh();
      return comparison;
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not generate comparison to parent.");
      return null;
    } finally {
      setGenerating(false);
    }
  }, [pipelineId, runId, refresh]);

  return { lineage, loading, generating, error, refresh, generateComparison };
}
