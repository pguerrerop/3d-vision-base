import type { FeatureDefinition } from "./api/client";

export type FeatureCatalogGroup = {
  sectionLabel: string;
  familyLabel: string;
  features: FeatureDefinition[];
};

function normalized(text: string | null | undefined): string {
  return String(text ?? "").trim().toLowerCase();
}

export function featureFamilyLabel(feature: FeatureDefinition): string {
  return String(feature.family_label ?? feature.semantic_group ?? feature.family ?? "Feature").trim() || "Feature";
}

export function featureSectionLabel(feature: FeatureDefinition): string {
  return feature.diagnostic_only ? "Diagnostic / engineering features" : "Stable ML/export features";
}

export function featureBadgeLabels(feature: FeatureDefinition): string[] {
  const badges: string[] = [];
  if (feature.stable_schema) badges.push("Stable ML");
  if (feature.diagnostic_only) badges.push("Diagnostic");
  if (feature.experimental) badges.push("Experimental");
  if (feature.higher_is_worse === true) badges.push("Higher worse");
  if (feature.higher_is_worse === false) badges.push("Higher better");
  return badges;
}

export function featureStudioStage(feature: FeatureDefinition | null | undefined): string | null {
  if (!feature) return null;
  const stage = String(feature.studio_stage ?? feature.source_stage ?? "").trim();
  return stage || null;
}

export function featureSearchText(feature: FeatureDefinition): string {
  return [
    feature.display_name,
    feature.feature_key,
    feature.family,
    feature.family_label,
    feature.semantic_group,
    feature.formula,
    feature.algorithm_summary,
    feature.interpretation,
    ...(feature.legacy_aliases ?? []),
  ]
    .map((item) => normalized(item))
    .filter(Boolean)
    .join(" ");
}

export function filterFeatureCatalog(features: FeatureDefinition[], search: string): FeatureDefinition[] {
  const query = normalized(search);
  if (!query) return features;
  return features.filter((feature) => featureSearchText(feature).includes(query));
}

export function featureOptionLabel(feature: FeatureDefinition): string {
  const badges = featureBadgeLabels(feature);
  const meta = [feature.feature_key, feature.unit || "no unit", ...badges].join(" · ");
  return `${feature.display_name} · ${meta}`;
}

export function groupFeatureCatalog(features: FeatureDefinition[]): FeatureCatalogGroup[] {
  const groups = new Map<string, FeatureCatalogGroup>();
  for (const feature of features) {
    const sectionLabel = featureSectionLabel(feature);
    const familyLabel = featureFamilyLabel(feature);
    const key = `${sectionLabel}::${familyLabel}`;
    const current = groups.get(key) ?? { sectionLabel, familyLabel, features: [] };
    current.features.push(feature);
    groups.set(key, current);
  }
  return [...groups.values()]
    .map((group) => ({
      ...group,
      features: [...group.features].sort((a, b) => a.display_name.localeCompare(b.display_name)),
    }))
    .sort((a, b) => {
      const sectionRank = (value: string) => (value === "Stable ML/export features" ? 0 : 1);
      if (a.sectionLabel !== b.sectionLabel) return sectionRank(a.sectionLabel) - sectionRank(b.sectionLabel);
      return a.familyLabel.localeCompare(b.familyLabel);
    });
}
