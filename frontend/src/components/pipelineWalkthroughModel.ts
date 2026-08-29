import type { PipelineInfo, PipelineStageInfo, ProcessingUnitDefinition, TakeDetail } from "../api/client.js";
import { processingUnitsForPipeline, processingUnitsForStage, rootProcessingUnit } from "./processingUnitModel.js";
import {
  blobComponentModeLabel,
  blobSplitMethodLabel,
  referenceSurfaceModelLabel,
  resolveDetectReferenceRunState,
  supportSelectionMethodLabel,
  suppressionMaskPolicyLabel,
  type ArtifactRelevance,
} from "./referenceArtifactRelevance.js";
import { canonicalStageId, stageSemanticDefinition } from "./stageSemantics.js";
import { resolveStageSubstagePlan, type ArtifactIoRole, type ResolvedStageSubstage } from "./stageSubstagePlan.js";
import { artifactList, artifactsForStage, type StudioArtifact } from "./studioWorkspaceModel.js";

export type WalkthroughFeedsInto = { artifactId: string; label: string };

export type WalkthroughArtifactInput = {
  id: string;
  label: string;
  artifactId?: string | null;
  description?: string | null;
  usage?: string | null;
  required: boolean;
};

export type WalkthroughArtifactProvenance = {
  calculation: string;
  calculatedBy: string;
  method: string;
  methodSteps: string[];
  tuneUnitId: string;
  inputs: WalkthroughArtifactInput[];
  parameters: WalkthroughParameter[];
};

export type WalkthroughArtifact = {
  artifactId: string;
  label: string;
  relevance: ArtifactRelevance;
  reason?: string;
  ioRole?: ArtifactIoRole;
  artifact: StudioArtifact | null;
  // Populated only for role="diagnostic" artifacts whose underlying data is verified (by reading
  // the pipeline source, not inferred) to determine another artifact's value -- as opposed to a
  // diagnostic that's purely a terminal report/visualization of decisions already finalized
  // elsewhere. Empty/undefined does not mean "unverified"; it means "checked, and it's a dead end."
  feedsInto?: WalkthroughFeedsInto[];
  provenance?: WalkthroughArtifactProvenance;
};

export type WalkthroughParameter = {
  id: string;
  label: string;
  value: unknown;
  formattedValue: string | null;
  default: unknown;
  unit?: string | null;
  advanced: boolean;
  relevant: boolean;
  relevanceReason?: string;
  tuningHint?: string | null;
  affects?: string[];
};

export type WalkthroughParameterGroup = {
  id: string;
  label: string;
  description?: string | null;
  affects?: string[];
  parameters: WalkthroughParameter[];
};

export type WalkthroughTuningHint = { condition: string; actions: string[] };

export type WalkthroughSubstage = {
  unitId: string;
  substageId: string;
  label: string;
  description?: string;
  relevance: ArtifactRelevance;
  artifacts: WalkthroughArtifact[];
  parameterGroups: WalkthroughParameterGroup[];
  ungroupedParameters: WalkthroughParameter[];
  downstreamEffects: string[];
  tuningHints: WalkthroughTuningHint[];
  helpMarkdown?: string | null;
};

export type WalkthroughStage = {
  stageId: string;
  label: string;
  description?: string;
  strategySummary?: Array<{ label: string; value: string }>;
  substages: WalkthroughSubstage[];
};

// supportingArtifacts' category is derived from real artifact presence plus unit activity.
// views' category can *also* mean "no legacy semantic-view-registry entry happens to exist for
// this view id" (see resolveStageViewForUnit's `missing` flag) -- an unrelated, purely
// backward-compatibility concept that has nothing to do with whether the artifact was actually
// produced for this take. Prefer the artifact-presence signal; only fall back to views' inactive/
// legacy signal (not its "missing", which is the overloaded one) when no artifacts are declared.
function substageRelevance(substage: ResolvedStageSubstage): ArtifactRelevance {
  if (substage.supportingArtifacts.length > 0) {
    const allInactive = substage.supportingArtifacts.every((entry) => entry.category === "inactive_strategy" || entry.category === "legacy");
    if (allInactive) return substage.supportingArtifacts[0]!.category;
    const allMissing = substage.supportingArtifacts.every((entry) => entry.category === "missing");
    if (allMissing) return "missing";
    return "active";
  }
  if (substage.views.length > 0) {
    const allInactive = substage.views.every((entry) => entry.category === "inactive_strategy" || entry.category === "legacy");
    if (allInactive) return substage.views[0]!.category;
  }
  return "active";
}

function formatParameterValue(value: unknown, unit?: string | null): string | null {
  if (value == null || value === "") return null;
  if (typeof value === "number") return unit ? `${value} ${unit}` : String(value);
  if (typeof value === "boolean") return value ? "On" : "Off";
  return String(value).replace(/_/g, " ");
}

function parameterRelevance(
  param: ProcessingUnitDefinition["parameters"][number],
  values: Record<string, unknown>,
): { relevant: boolean; reason?: string } {
  if (!param.active_when) return { relevant: true };
  for (const [dep, expected] of Object.entries(param.active_when)) {
    if (values[dep] !== expected) {
      return { relevant: false, reason: `Requires ${dep.replace(/_/g, " ")} = ${String(expected).replace(/_/g, " ")}` };
    }
  }
  return { relevant: true };
}

function buildParameters(
  unit: ProcessingUnitDefinition,
  values: Record<string, unknown>,
): { groups: WalkthroughParameterGroup[]; ungrouped: WalkthroughParameter[] } {
  const toWalkthroughParameter = (param: ProcessingUnitDefinition["parameters"][number]): WalkthroughParameter => {
    const value = values[param.id] !== undefined ? values[param.id] : param.default;
    const { relevant, reason } = parameterRelevance(param, values);
    return {
      id: param.id,
      label: param.label,
      value,
      formattedValue: formatParameterValue(value, param.unit),
      default: param.default,
      unit: param.unit,
      advanced: Boolean(param.advanced),
      relevant,
      relevanceReason: reason,
      tuningHint: param.tuning_hint,
      affects: param.affects,
    };
  };

  const claimed = new Set<string>();
  const groups: WalkthroughParameterGroup[] = (unit.parameter_groups ?? [])
    .map((group) => {
      const keys = (group.param_keys ?? []).filter((key) => unit.parameters.some((p) => p.id === key));
      for (const key of keys) claimed.add(key);
      return {
        id: group.id,
        label: group.label,
        description: group.description,
        affects: group.affects,
        parameters: keys.map((key) => toWalkthroughParameter(unit.parameters.find((p) => p.id === key)!)),
      };
    })
    .filter((group) => group.parameters.length > 0 || group.description || (group.affects?.length ?? 0) > 0);
  const ungrouped = unit.parameters.filter((p) => !claimed.has(p.id)).map(toWalkthroughParameter);
  return { groups, ungrouped };
}

function normalizeProvenanceKey(value: string): string {
  return value.trim().toLowerCase().replace(/[^a-z0-9]+/g, "_").replace(/^_+|_+$/g, "");
}

function artifactCalculation(
  unit: ProcessingUnitDefinition,
  artifact: WalkthroughArtifact,
): string {
  const declared = unit.artifacts.find((entry) => entry.artifact_id === artifact.artifactId);
  if (declared?.description) return declared.description;

  const artifactId = artifact.artifactId;
  const explicit: Record<string, string> = {
    low_gradient_components_overlay: "Connected components in the low-gradient candidate mask are assigned distinct colors. For the background-and-stripes strategy, this visualizes the component classification used to separate belt background, stripes, and unknown regions.",
    low_gradient_components: "The connected-component labels from the low-gradient candidate mask are summarized as one record per component, including the values used by the background-and-stripes classifier.",
    low_gradient_blob_components_overlay: "Connected components are labeled from the low-gradient mask using the configured pixel connectivity, then rendered with one color per component. This is the XY-only blob-branch intermediate.",
    low_gradient_blob_id_mask: "Each low-gradient pixel is assigned the integer ID of its XY-connected component; background pixels remain zero. Downstream blob splitting and clustering consume these component IDs.",
    height_aware_blob_components_overlay: "Low-gradient pixels are connected only when neighboring heights meet the configured Z-tolerance and slope rules, then each retained component is rendered in a distinct color.",
    height_aware_blob_id_mask: "Each height-consistent low-gradient component receives an integer ID. Neighboring pixels separated by the configured Z-tolerance or slope rule are assigned different components.",
    depth_gradient_magnitude: "For each valid heightmap pixel, the local depth-gradient magnitude is computed after the configured smoothing. The image renders that scalar field.",
    low_gradient_mask: "Valid pixels whose depth-gradient magnitude passes the configured threshold are retained, then the configured morphology and minimum-area cleanup are applied.",
    flat_candidate_mask: "Pixels are retained when they meet the early height-gate rules relative to the candidate reference height; this prunes implausible raised regions before later support selection.",
    plane_residual_heatmap: "For each evaluated pixel, the signed distance from the fitted reference plane is computed and rendered as a heatmap.",
    final_plane_inlier_mask: "Pixels whose residual to the final fitted reference plane meets the inlier rule are retained as the plane-fit support.",
    selected_reference_support_mask: "The strategy-selected support is refined and stripe-suppressed, then emitted as the final binary reference-support mask.",
    reference_suppression_mask: "The final selected reference support is converted into the mask used to suppress belt/reference pixels in downstream object segmentation.",
  };
  if (explicit[artifactId]) return explicit[artifactId]!;

  if (artifactId.endsWith("_id_mask")) {
    return `Each retained pixel is assigned the integer label of the ${unit.label.toLowerCase()} component it belongs to; non-member pixels are zero.`;
  }
  if (artifactId.endsWith("_mask")) {
    return `This binary mask retains the pixels that pass the ${unit.label.toLowerCase()} decision rules and clears all other pixels.`;
  }
  if (artifactId.endsWith("_overlay")) {
    return `This display-only overlay renders the per-pixel result of ${unit.label.toLowerCase()} so its component or decision regions can be inspected.`;
  }
  if (artifactId.includes("histogram") || artifactId.includes("plot")) {
    return `This diagnostic plots the distribution or score values computed by ${unit.label.toLowerCase()} before its selection decision.`;
  }
  if (artifact.artifact?.kind === "json" || artifactId.endsWith("_debug") || artifactId.endsWith("_summary")) {
    return `This structured diagnostic serializes the intermediate values and decisions produced by ${unit.label.toLowerCase()}.`;
  }
  return `This artifact is emitted directly by ${unit.label.toLowerCase()} from the declared inputs and the parameter values shown below.`;
}

const BG_AND_STRIPES_COMPONENT_ARTIFACTS = new Set([
  "low_gradient_components_overlay",
  "low_gradient_components",
  "belt_bg_components",
]);

function provenanceUnitForArtifact(
  displayedUnit: ProcessingUnitDefinition,
  artifact: WalkthroughArtifact,
  units: ProcessingUnitDefinition[],
): ProcessingUnitDefinition {
  if (!BG_AND_STRIPES_COMPONENT_ARTIFACTS.has(artifact.artifactId)) return displayedUnit;
  return units.find((unit) => unit.id === "detect_belt_plane.stripe_filter") ?? displayedUnit;
}

function artifactInputs(
  unit: ProcessingUnitDefinition,
  artifact: WalkthroughArtifact,
): WalkthroughArtifactInput[] {
  if (BG_AND_STRIPES_COMPONENT_ARTIFACTS.has(artifact.artifactId)) {
    return [
      {
        id: "low_gradient_mask",
        label: "Low-gradient mask",
        artifactId: "low_gradient_mask",
        description: "Binary candidate mask of calm surface pixels that survived the gradient gate.",
        usage: "Defines which connected low-gradient regions are eligible for belt-background and stripe classification.",
        required: true,
      },
      {
        id: "plateau",
        label: "Selected belt-background plateau",
        artifactId: "background_selected_plateau_mask",
        description: "The plateau already chosen as the likely belt/background height band.",
        usage: "Anchors the component features against the selected belt-background reference instead of the whole ROI.",
        required: true,
      },
      {
        id: "valid_mask",
        label: "Valid sensor mask",
        artifactId: "valid_mask",
        description: "Pixels with usable sensor readings inside the active heightmap domain.",
        usage: "Prevents invalid or missing-height pixels from participating in the component measurements.",
        required: true,
      },
    ];
  }
  return unit.inputs.map((input) => ({
    id: input.id,
    label: input.label,
    artifactId: input.artifact_id,
    description: input.description,
    usage: artifactInputUsage(unit, artifact, input.artifact_id ?? input.id, input.label),
    required: input.required !== false,
  }));
}

function artifactInputUsage(
  unit: ProcessingUnitDefinition,
  artifact: WalkthroughArtifact,
  inputId: string,
  inputLabel: string,
): string | null {
  const explicit: Record<string, string> = {
    low_gradient_mask: "Acts as the candidate search domain for later component, plateau, or stripe decisions.",
    depth_gradient_magnitude: "Provides the per-pixel roughness signal that this step thresholds into a calmer support candidate region.",
    valid_mask: "Limits the calculation to pixels with valid height data.",
    background_selected_plateau_mask: "Supplies the belt/background reference region used to measure above-belt or stripe-like structure.",
    selected_blob_cluster_pre_refine_mask: "Provides the pre-refinement support region that later cleanup or stripe suppression trims.",
    selected_blob_cluster_refined_mask: "Provides the refined support mask that stripe suppression or final selection starts from.",
    stripe_filtered_reference_support_mask: "Supplies the post-stripe support mask that later stages hand to model fitting or final support selection.",
    selected_reference_support_mask: "Acts as the chosen logical support mask that downstream model fitting or suppression policy consumes.",
    reference_model_support_mask: "Supplies the model-accepted support mask used to derive downstream suppression or fit diagnostics.",
  };
  if (explicit[inputId]) return explicit[inputId]!;

  if (artifact.artifactId.endsWith("_overlay")) {
    return `${inputLabel} is visualized or color-coded here so the method's internal regions can be inspected.`;
  }
  if (artifact.artifactId.endsWith("_mask")) {
    return `${inputLabel} is one of the upstream masks this method combines or filters to decide which pixels stay active.`;
  }
  if (artifact.artifact?.kind === "json") {
    return `${inputLabel} contributes measurements or decisions that are summarized in this structured diagnostic output.`;
  }
  return `${inputLabel} is consumed by ${unit.label.toLowerCase()} while producing this artifact.`;
}

function artifactParameters(
  unit: ProcessingUnitDefinition,
  artifact: WalkthroughArtifact,
  parameters: WalkthroughParameter[],
): WalkthroughParameter[] {
  if (!BG_AND_STRIPES_COMPONENT_ARTIFACTS.has(artifact.artifactId)) return parameters;
  const componentClassification = unit.parameter_groups?.find((group) => group.id === "stripe_component_classification");
  const keys = new Set(componentClassification?.param_keys ?? []);
  const selected = parameters.filter((parameter) => keys.has(parameter.id));
  return selected.length > 0 ? selected : parameters;
}

function artifactMethodSteps(
  unit: ProcessingUnitDefinition,
  artifact: WalkthroughArtifact,
  inputs: WalkthroughArtifactInput[],
  parameters: WalkthroughParameter[],
): string[] {
  const artifactId = artifact.artifactId;
  const explicit: Record<string, string[]> = {
    low_gradient_mask: [
      "Compute local depth-gradient magnitude over valid pixels using the configured smoothing and invalid-neighbor handling.",
      "Threshold that gradient field with the chosen percentile, Otsu, or fixed rule to keep calmer surface regions.",
      "Optionally apply morphology, hole filling, and minimum-area cleanup before emitting the final candidate mask.",
    ],
    low_gradient_components_overlay: [
      "Take the low-gradient candidate mask and form connected components over the surviving pixels.",
      "Measure each component against the selected belt-background plateau and valid-sensor domain.",
      "Classify each component as belt background, belt stripe, or unknown, then render those labels as a color overlay.",
    ],
    low_gradient_components: [
      "Take the low-gradient candidate mask and form connected components over the surviving pixels.",
      "Measure each component against the selected belt-background plateau and valid-sensor domain.",
      "Write one summary record per component with the features and class decisions used by the background-and-stripes strategy.",
    ],
    belt_stripes_mask: [
      "Start from the incoming refined support region and the configured stripe-filter search scope.",
      "Detect stripe evidence with the stripe filter method, combining altitude and optional shape-based passes as configured.",
      "Keep only the pixels classified as belt stripes and emit their union as the final stripe mask.",
    ],
    surface_suppression_mask: [
      "Collect the belt-background support and the accepted belt-stripe detections from the stripe filter.",
      "Merge those accepted surface regions into the explicit downstream suppression mask.",
      "Emit the final surface-suppression handoff used by later object segmentation logic.",
    ],
    selected_reference_support_mask: [
      "Start from the strategy-selected support candidate produced upstream.",
      "Apply the configured refinement, fallback, and stripe-suppression decisions that are active for this run.",
      "Emit the final logical reference-support mask handed to downstream model fitting and suppression policy.",
    ],
    reference_suppression_mask: [
      "Take the selected support and reference-model support artifacts resolved upstream.",
      "Apply the configured suppression-mask policy to choose which support source should be suppressed downstream.",
      "Emit the final suppression mask handed to segmentation.",
    ],
  };
  if (explicit[artifactId]) return explicit[artifactId]!;

  const steps: string[] = [];
  if (inputs.length > 0) {
    steps.push(`Consume ${inputs.map((input) => input.label).join(", ")} as the upstream inputs for ${unit.label.toLowerCase()}.`);
  }
  steps.push(artifactCalculation(unit, artifact));
  if (parameters.length > 0) {
    steps.push(`Apply the ${parameters.length === 1 ? "parameter" : "parameters"} shown here to control thresholds, cleanup, and acceptance rules for this output.`);
  }
  return steps;
}

function artifactProvenance(
  unit: ProcessingUnitDefinition,
  artifact: WalkthroughArtifact,
  parameters: WalkthroughParameter[],
): WalkthroughArtifactProvenance {
  const artifactKeys = new Set([artifact.artifactId, artifact.label].map(normalizeProvenanceKey));
  const directlyAffecting = parameters.filter((param) =>
    (param.affects ?? []).some((effect) => artifactKeys.has(normalizeProvenanceKey(effect))),
  );
  const selectedParameters = directlyAffecting.length > 0 ? directlyAffecting : artifactParameters(unit, artifact, parameters);
  const inputs = artifactInputs(unit, artifact);
  return {
    calculation: artifactCalculation(unit, artifact),
    calculatedBy: unit.label,
    method: unit.description,
    methodSteps: artifactMethodSteps(unit, artifact, inputs, selectedParameters),
    tuneUnitId: unit.id,
    inputs,
    // Some older contract entries name a broad outcome (for example "Selected support") rather
    // than each emitted artifact. In that case, showing the unit's controls is more useful than
    // pretending the artifact has no adjustable inputs.
    parameters: selectedParameters,
  };
}

// Artifact files are written with a real per-artifact registration timestamp (microsecond
// precision, confirmed distinct and sequential in practice), which is a true production-order
// signal -- unlike the contract's artifacts=[...] declaration order, which is just narrative
// (input/intermediate/output/diagnostic grouping) and doesn't necessarily match the order the
// pipeline actually computed and wrote each file. Artifacts with no timestamp (never produced,
// i.e. "missing") keep their declared relative order and sort after every timestamped one.
//
// Compared as raw strings, not via Date.parse/getTime(): real timestamps within one substage are
// often only microseconds apart (e.g. "...503679Z" vs "...503747Z", the same millisecond) --
// Date.parse silently truncates to millisecond precision and collapses them to equal values,
// which would make this sort a no-op for exactly the common case it exists to handle. These
// timestamps are fixed-width, zero-padded ISO-8601 UTC strings, so plain lexicographic string
// comparison already equals chronological order at full precision.
function sortArtifactsByProductionOrder(artifacts: WalkthroughArtifact[]): WalkthroughArtifact[] {
  return artifacts
    .map((entry, index) => ({ entry, index, time: entry.artifact?.created_at ?? null }))
    .sort((a, b) => {
      if (a.time != null && b.time != null) return a.time < b.time ? -1 : a.time > b.time ? 1 : a.index - b.index;
      if (a.time != null) return -1;
      if (b.time != null) return 1;
      return a.index - b.index;
    })
    .map(({ entry }) => entry);
}

function buildUnitEntry(
  unit: ProcessingUnitDefinition,
  substageId: string,
  relevance: ArtifactRelevance,
  artifacts: WalkthroughArtifact[],
  values: Record<string, unknown>,
  provenanceUnits: ProcessingUnitDefinition[] = [unit],
): WalkthroughSubstage {
  const { groups, ungrouped } = buildParameters(unit, values);
  const parameters = groups.flatMap((group) => group.parameters).concat(ungrouped);
  return {
    unitId: unit.id,
    substageId,
    label: unit.label,
    description: unit.description,
    relevance,
    artifacts: sortArtifactsByProductionOrder(artifacts).map((artifact) => {
      const provenanceUnit = provenanceUnitForArtifact(unit, artifact, provenanceUnits);
      const provenanceParameters = provenanceUnit === unit
        ? parameters
        : (() => {
          const { groups: provenanceGroups, ungrouped: provenanceUngrouped } = buildParameters(provenanceUnit, values);
          return provenanceGroups.flatMap((group) => group.parameters).concat(provenanceUngrouped);
        })();
      return {
        ...artifact,
        provenance: artifactProvenance(provenanceUnit, artifact, provenanceParameters),
      };
    }),
    parameterGroups: groups,
    ungroupedParameters: ungrouped,
    downstreamEffects: unit.downstream_effects ?? [],
    tuningHints: unit.tuning_hints ?? [],
    helpMarkdown: unit.help_markdown,
  };
}

function allStageParams(detail: TakeDetail | null): Record<string, unknown> {
  const result = (detail?.result as unknown as Record<string, unknown> | null) ?? {};
  const raw = result["stage_params"];
  return raw && typeof raw === "object" ? (raw as Record<string, unknown>) : {};
}

// stage_params is namespaced per stage (e.g. {"detect_belt_plane": {...}, "known_object_25d": {...}}),
// keyed by each stage's own parameter_schema.runtime_stage_params_key -- which does not always match
// the stage's own id (measurement_diagnostics's schema is keyed "known_object_25d"). Falls back to the
// stage id itself for stages with no parameter_schema (nothing gated there anyway).
function valuesForStage(stage: PipelineStageInfo, stageParams: Record<string, unknown>): Record<string, unknown> {
  const schema = stage.parameter_schema as Record<string, unknown> | undefined;
  const key = (schema?.["runtime_stage_params_key"] as string | undefined) ?? stage.id;
  const scoped = stageParams[key];
  return scoped && typeof scoped === "object" ? (scoped as Record<string, unknown>) : {};
}

function strategySummaryForStage(
  stageId: string,
  detail: TakeDetail | null,
  stageArtifacts: StudioArtifact[],
): WalkthroughStage["strategySummary"] {
  if (canonicalStageId(stageId) !== "detect_belt_plane") return undefined;
  const state = resolveDetectReferenceRunState(detail, stageArtifacts, null);
  const summary = [
    { label: "Strategy", value: supportSelectionMethodLabel(state.supportSelectionMethod) },
    { label: "Reference model", value: referenceSurfaceModelLabel(state.referenceSurfaceModel) },
    { label: "Suppression", value: suppressionMaskPolicyLabel(state.suppressionMaskPolicy) },
  ];
  if (state.strategy === "low_gradient_blob_height_clusters") {
    summary.splice(
      1,
      0,
      { label: "Components", value: blobComponentModeLabel(state.componentMode) },
      { label: "Split", value: blobSplitMethodLabel(state.splitMethod) },
    );
  }
  return summary.filter((entry) => entry.value !== "Unknown");
}

type ArtifactGraphNode = { role: string; unitId: string; feedsInto: string[] };

// Whether a diagnostic artifact "really" leaves its substage is a transitive-reachability question,
// not a first-hop one: a diagnostic can point at another diagnostic that itself goes nowhere (still
// a dead end -- must not be marked OUT), or at another diagnostic that *does* eventually reach a real
// final/input artifact in a different unit (genuinely OUT, even though the immediate hop looks the
// same). Memoized per (originUnitId, artifactId) since multiple diagnostics in one substage often
// converge on the same downstream node -- cheap either way at this graph's size (tens of nodes,
// mostly 0-1 edges each), but avoids repeating the same walk.
function reachesExternalOutput(
  artifactId: string,
  originUnitId: string,
  graph: Map<string, ArtifactGraphNode>,
  memo: Map<string, boolean>,
  visiting: Set<string> = new Set(),
): boolean {
  const cacheKey = `${originUnitId}::${artifactId}`;
  const cached = memo.get(cacheKey);
  if (cached !== undefined) return cached;
  if (visiting.has(artifactId)) return false;
  visiting.add(artifactId);
  let result = false;
  for (const nextId of graph.get(artifactId)?.feedsInto ?? []) {
    const next = graph.get(nextId);
    if (!next) continue;
    if (next.unitId !== originUnitId && (next.role === "final" || next.role === "input")) {
      result = true;
      break;
    }
    if (reachesExternalOutput(nextId, originUnitId, graph, memo, visiting)) {
      result = true;
      break;
    }
  }
  visiting.delete(artifactId);
  memo.set(cacheKey, result);
  return result;
}

/**
 * Walks every stage of the pipeline for a given take, producing a flat, relevance-tagged
 * report of every substage's artifacts and parameters -- the same contract data that drives
 * the live Tune/Artifacts panels, just assembled for a single all-in-one printable view.
 */
export function buildPipelineWalkthrough(pipeline: PipelineInfo | null, detail: TakeDetail | null): WalkthroughStage[] {
  if (!pipeline) return [];
  const stageParams = allStageParams(detail);
  const takeArtifacts = artifactList(detail);
  // feeds_into targets can live on any unit in any stage (e.g. a detect_belt_plane diagnostic
  // feeding Segment's final mask), so both the id -> label lookup and the reachability graph have
  // to span the whole pipeline, not just the current stage's own units.
  const allArtifactLabels = new Map<string, string>();
  const artifactGraph = new Map<string, ArtifactGraphNode>();
  for (const unit of processingUnitsForPipeline(pipeline, detail)) {
    for (const artifact of unit.artifacts) {
      if (!allArtifactLabels.has(artifact.artifact_id)) allArtifactLabels.set(artifact.artifact_id, artifact.label);
      if (!artifactGraph.has(artifact.artifact_id)) {
        artifactGraph.set(artifact.artifact_id, { role: artifact.role, unitId: unit.id, feedsInto: artifact.feeds_into ?? [] });
      }
    }
  }
  const reachabilityMemo = new Map<string, boolean>();
  const resolveFeedsInto = (ids: string[] | undefined): WalkthroughFeedsInto[] | undefined => {
    if (!ids || ids.length === 0) return undefined;
    return ids.map((artifactId) => ({ artifactId, label: allArtifactLabels.get(artifactId) ?? artifactId.replace(/_/g, " ") }));
  };
  // Unifies the two ways an artifact earns "OUT": a declared role="final"/"input" (the substage's own
  // intentional hand-off), or a role="diagnostic" whose feeds_into chain is verified to transitively
  // reach a real final/input artifact outside this unit -- data that leaves the substage in practice
  // even though the contract never labeled it a deliverable. A diagnostic whose chain only reaches
  // other diagnostics (or nothing) does not qualify and stays plain DEBUG.
  const resolveIoRole = (role: string, unitId: string, artifactId: string): ArtifactIoRole | undefined => {
    if (role === "input") return "input";
    if (role === "final") return "output";
    if (role === "diagnostic" && reachesExternalOutput(artifactId, unitId, artifactGraph, reachabilityMemo)) return "output";
    return undefined;
  };

  return pipeline.stages.map((stage): WalkthroughStage => {
    const values = valuesForStage(stage, stageParams);
    const canonical = canonicalStageId(stage.id, stage.display_name);
    const semantic = stageSemanticDefinition(canonical);
    const stageArtifacts = artifactsForStage(detail, stage.id);
    const resolution = resolveStageSubstagePlan(stage.id, semantic, pipeline, detail, stageArtifacts, takeArtifacts);
    const units = processingUnitsForStage(pipeline, detail, canonical);

    const substages: WalkthroughSubstage[] = [];
    const root = rootProcessingUnit(units);
    if (root) {
      const presentIds = new Set(stageArtifacts.map((item) => item.artifact_id).concat(takeArtifacts.map((item) => item.artifact_id)));
      const rootArtifacts: WalkthroughArtifact[] = root.artifacts.map((artifact) => {
        const found = stageArtifacts.find((item) => item.artifact_id === artifact.artifact_id)
          ?? takeArtifacts.find((item) => item.artifact_id === artifact.artifact_id)
          ?? null;
        return {
          artifactId: artifact.artifact_id,
          label: artifact.label,
          relevance: found ? (artifact.role === "diagnostic" ? "debug" : "active") : "missing",
          reason: artifact.description ?? undefined,
          ioRole: resolveIoRole(artifact.role, root.id, artifact.artifact_id),
          artifact: found,
          feedsInto: resolveFeedsInto(artifact.feeds_into),
        };
      });
      const rootRelevance: ArtifactRelevance = root.artifacts.length > 0 && !root.artifacts.some((a) => presentIds.has(a.artifact_id))
        ? "missing"
        : "active";
      if (root.parameters.length > 0 || rootArtifacts.length > 0) {
        substages.push(buildUnitEntry(root, "strategy_path", rootRelevance, rootArtifacts, values, units));
      }
    }

    for (const resolved of resolution.substages) {
      const unit = units.find((item) => item.id === resolved.unitId);
      const artifacts: WalkthroughArtifact[] = resolved.supportingArtifacts.map((entry) => {
        const declared = unit?.artifacts.find((a) => a.artifact_id === entry.artifactId);
        return {
          artifactId: entry.artifactId,
          label: entry.artifact?.title ?? entry.artifactId,
          relevance: entry.category,
          reason: entry.reason,
          ioRole: unit && declared ? resolveIoRole(declared.role, unit.id, declared.artifact_id) : entry.ioRole,
          artifact: entry.artifact,
          feedsInto: resolveFeedsInto(declared?.feeds_into),
        };
      });
      const relevance = substageRelevance(resolved);
      if (unit) {
        substages.push(buildUnitEntry(unit, resolved.substageId, relevance, artifacts, values, units));
      } else {
        substages.push({
          unitId: resolved.unitId ?? resolved.substageId,
          substageId: resolved.substageId,
          label: resolved.label,
          description: resolved.description,
          relevance,
          artifacts,
          parameterGroups: [],
          ungroupedParameters: [],
          downstreamEffects: [],
          tuningHints: [],
        });
      }
    }

    return {
      stageId: stage.id,
      label: stage.display_name,
      description: stage.description,
      strategySummary: strategySummaryForStage(stage.id, detail, stageArtifacts),
      substages,
    };
  });
}
