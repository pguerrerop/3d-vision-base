export type StudioDeepLinkParams = {
  take_id?: string | null;
  pipeline_id?: string | null;
  run_id?: string | null;
  stage?: string | null;
  object_id?: string | null;
  physical_object_id?: string | null;
  feature?: string | null;
  dataset_id?: string | null;
  session_id?: string | null;
  graph_workspace?: string | null;
  unit_id?: string | null;
  comparison_id?: string | null;
  recipe_id?: string | null;
  tab?: string | null;
};

function cleaned(value: string | null | undefined): string | null {
  const next = String(value ?? "").trim();
  return next ? next : null;
}

export function buildStudioDeepLink(params: StudioDeepLinkParams): string {
  const qs = new URLSearchParams();
  const ordered: Array<keyof StudioDeepLinkParams> = [
    "take_id",
    "pipeline_id",
    "run_id",
    "recipe_id",
    "comparison_id",
    "unit_id",
    "tab",
    "graph_workspace",
    "stage",
    "object_id",
    "physical_object_id",
    "feature",
    "dataset_id",
    "session_id",
  ];
  for (const key of ordered) {
    const value = cleaned(params[key]);
    if (value) qs.set(key, value);
  }
  return `/studio${qs.toString() ? `?${qs.toString()}` : ""}`;
}

export function parseStudioDeepLink(search: string): StudioDeepLinkParams {
  const params = new URLSearchParams(search.startsWith("?") ? search : `?${search}`);
  return {
    take_id: cleaned(params.get("take_id")),
    pipeline_id: cleaned(params.get("pipeline_id")),
    run_id: cleaned(params.get("run_id")),
    stage: cleaned(params.get("stage")),
    object_id: cleaned(params.get("object_id")),
    physical_object_id: cleaned(params.get("physical_object_id")),
    feature: cleaned(params.get("feature")),
    dataset_id: cleaned(params.get("dataset_id")),
    session_id: cleaned(params.get("session_id")),
    graph_workspace: cleaned(params.get("graph_workspace")),
    unit_id: cleaned(params.get("unit_id")),
    comparison_id: cleaned(params.get("comparison_id")),
    recipe_id: cleaned(params.get("recipe_id")),
    tab: cleaned(params.get("tab")),
  };
}

export function graphWorkspaceEnabled(value: string | null | undefined): boolean {
  const normalized = String(value ?? "").trim().toLowerCase();
  return normalized === "1" || normalized === "true" || normalized === "yes";
}
