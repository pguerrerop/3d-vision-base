import { spawnSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";

/**
 * Warn when data/index.db no longer matches what is on disk.
 *
 * The catalog is pushed to by whoever writes, so a missed hook shows up as
 * drift rather than as an error. This runs when the dev services come up: it is
 * the cheap moment to notice, long before someone wonders why a take is absent
 * from a listing. It only ever prints — starting the stack must not depend on
 * the index being healthy.
 */
export function reportCatalogDrift({ repoRoot, python, dataDir }) {
  const database = path.join(dataDir, "index.db");
  if (!fs.existsSync(database)) {
    return { checked: false, reason: "no catalog yet" };
  }

  const result = spawnSync(
    python,
    [path.join(repoRoot, "scripts", "sensor_studio_cli.py"), "index", "status", "--data-dir", dataDir],
    { cwd: repoRoot, encoding: "utf-8", timeout: 30_000 },
  );

  if (result.error || typeof result.stdout !== "string" || !result.stdout.trim()) {
    return { checked: false, reason: "index status did not run" };
  }

  let status;
  try {
    status = JSON.parse(result.stdout);
  } catch {
    return { checked: false, reason: "index status returned unparseable output" };
  }

  const drift = Number(status.drift ?? 0);
  const staleProjection =
    String(status.indexed_projection_version ?? "") !== String(status.projection_version ?? "");
  const staleSchema = Number(status.schema_version ?? 0) !== Number(status.latest_schema_version ?? 0);

  if (drift !== 0) {
    console.warn(
      `[catalog] drift: ${status.takes_on_disk} takes on disk, ${status.counts?.take_index} indexed.` +
        " Run: python scripts/sensor_studio_cli.py index rebuild --data-dir data",
    );
  }
  if (staleProjection || staleSchema) {
    console.warn(
      "[catalog] the index predates the current schema or projection." +
        " Run: python scripts/sensor_studio_cli.py index rebuild --data-dir data --full",
    );
  }
  return { checked: true, drift, staleProjection, staleSchema };
}
