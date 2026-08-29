import test from "node:test";
import assert from "node:assert/strict";
import {
  detectReferenceArtifactRoleHelpKey,
  detectReferenceDisplayRoleHelpKey,
  detectReferenceGroupHelpKey,
  detectReferenceLineageSummaryHelpKey,
  detectReferenceParamHelpKey,
  detectReferenceSubstageHelpKey,
  detectReferenceViewHelpKey,
  resolveStudioHelpEntry,
} from "../src/components/studioHelp.ts";

test("studio help registry resolves expected detect-reference entries", () => {
  assert.equal(resolveStudioHelpEntry("detect_reference_surface.process.gradient_low_gradient")?.title, "Gradient / low-gradient");
  assert.equal(resolveStudioHelpEntry("detect_reference_surface.artifact.selected_surface")?.title, "Selected support");
  assert.equal(resolveStudioHelpEntry("detect_reference_surface.param.mad_k")?.title, "MAD k");
  assert.equal(resolveStudioHelpEntry("detect_reference_surface.artifact.reference_model_support")?.title, "Reference model support");
  assert.equal(resolveStudioHelpEntry("detect_reference_surface.param.blob_split_height_border_threshold_mode")?.optionDetails?.length, 3);
  assert.equal(resolveStudioHelpEntry("detect_reference_surface.group.belt_stripe_suppression")?.title, "Belt stripe suppression");
  assert.match(
    resolveStudioHelpEntry("detect_reference_surface.group.belt_stripe_suppression")?.details ?? "",
    /top-hat/i,
  );
});

test("help key resolvers map substages, views, and params to stable keys", () => {
  assert.equal(detectReferenceSubstageHelpKey("cluster_scoring"), "detect_reference_surface.process.cluster_scoring");
  assert.equal(detectReferenceSubstageHelpKey("stripes"), "detect_reference_surface.group.belt_stripe_suppression");
  assert.equal(detectReferenceGroupHelpKey("Belt stripe suppression"), "detect_reference_surface.group.belt_stripe_suppression");
  assert.equal(detectReferenceGroupHelpKey("stripe_input"), "detect_reference_surface.group.stripe_input");
  assert.equal(detectReferenceViewHelpKey("selected_blob_cluster"), "detect_reference_surface.artifact.selected_blob_cluster");
  assert.equal(detectReferenceParamHelpKey("blob_cluster_refine_mad_k"), "detect_reference_surface.param.mad_k");
  assert.equal(detectReferenceParamHelpKey("belt_stripe_filter_enabled"), "detect_reference_surface.param.belt_stripe_filter_enabled");
  assert.equal(detectReferenceSubstageHelpKey("fragment_merge"), "detect_reference_surface.process.fragment_merge");
  assert.equal(detectReferenceViewHelpKey("height_border_fragments"), "detect_reference_surface.artifact.height_border_fragments");
  assert.equal(detectReferenceViewHelpKey("reference_model_support"), "detect_reference_surface.artifact.reference_model_support");
  assert.equal(detectReferenceParamHelpKey("blob_split_height_border_percentile"), "detect_reference_surface.param.blob_split_height_border_percentile");
  assert.equal(detectReferenceArtifactRoleHelpKey("suppression_mask"), "detect_reference_surface.role.suppression_mask");
  assert.equal(detectReferenceDisplayRoleHelpKey("fit_support"), "detect_reference_surface.role.fit_support");
  assert.equal(detectReferenceLineageSummaryHelpKey("Selected support"), "detect_reference_surface.artifact.selected_surface");
});
