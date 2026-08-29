import type { FeatureAnalyticsQuery } from "./api/client";

export function featureAnalyticsQueryToSearchParams(query: FeatureAnalyticsQuery & { feature_selection?: string[] }): URLSearchParams {
  const qs = new URLSearchParams();
  const appendMany = (key: string, values?: string[]) => {
    for (const value of values ?? []) {
      const text = String(value ?? "").trim();
      if (text) qs.append(key, text);
    }
  };
  if (query.dataset_id) qs.set("dataset_id", query.dataset_id);
  if (query.ml_set_id) qs.set("ml_set_id", query.ml_set_id);
  if (query.session_id) qs.set("session_id", query.session_id);
  appendMany("physical_object_ids", query.physical_object_ids);
  appendMany("take_ids", query.take_ids);
  appendMany("labeled_superclasses", query.labeled_superclasses);
  appendMany("processed_superclasses", query.processed_superclasses);
  appendMany("raw_labels", query.raw_labels);
  appendMany("normalized_classes", query.normalized_classes);
  appendMany("processed_classes", query.processed_classes);
  appendMany("superclasses", query.superclasses);
  appendMany("validation_status", query.validation_status);
  appendMany("split", query.split);
  if (query.pipeline_id) qs.set("pipeline_id", query.pipeline_id);
  if (query.calibration_id) qs.set("calibration_id", query.calibration_id);
  if (query.date_from) qs.set("date_from", query.date_from);
  if (query.date_to) qs.set("date_to", query.date_to);
  appendMany("feature_selection", query.feature_selection);
  return qs;
}

export function featureAnalyticsFeatureCatalogScope(query: FeatureAnalyticsQuery): FeatureAnalyticsQuery {
  return {
    dataset_id: query.dataset_id,
    ml_set_id: query.ml_set_id,
    session_id: query.session_id,
    pipeline_id: query.pipeline_id,
    calibration_id: query.calibration_id,
    date_from: query.date_from,
    date_to: query.date_to,
  };
}
