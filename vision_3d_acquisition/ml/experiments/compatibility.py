from __future__ import annotations

from typing import Any

from vision_3d_acquisition.ml.experiments.contracts import ExperimentConfig
from vision_3d_acquisition.ml.features.dataset import FeatureDataset


def validate_experiment_compatibility(dataset: FeatureDataset, config: ExperimentConfig) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    if dataset.dataset_id != config.dataset_id:
        errors.append("dataset_id_mismatch")
    if dataset.feature_schema.feature_schema_version != config.feature_schema_version:
        errors.append("feature_schema_version_mismatch")

    groups = dataset.get_feature_groups()
    selected_groups = list(config.feature_selection.get("include_groups") or groups.keys())
    for group in selected_groups:
        if group not in groups:
            errors.append(f"unknown_feature_group:{group}")
    excluded_groups = list(config.feature_selection.get("exclude_groups") or [])
    for group in excluded_groups:
        if group not in groups:
            warnings.append(f"exclude_unknown_group:{group}")

    explicit_features = list(config.feature_selection.get("include_features") or [])
    if explicit_features:
        names = set(dataset.get_feature_names())
        missing = sorted([name for name in explicit_features if name not in names])
        if missing:
            errors.append(f"missing_selected_features:{','.join(missing)}")

    normalization_mode = str(config.normalization.get("mode") or "none")
    if normalization_mode not in {"none", "zscore", "robust_zscore", "minmax"}:
        errors.append(f"unsupported_normalization_mode:{normalization_mode}")

    if not any(sample.label or sample.metadata.get("synthetic_object_type") for sample in dataset.samples):
        errors.append("dataset_has_no_labels")

    dataset_validation = dataset.validate()
    if not dataset_validation.get("ok"):
        warnings.append("dataset_validation_has_errors")

    return {
        "ok": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "summary": {
            "dataset_id": dataset.dataset_id,
            "feature_schema_version": dataset.feature_schema.feature_schema_version,
            "sample_count": len(dataset.samples),
            "selected_group_count": len(selected_groups),
        },
    }
