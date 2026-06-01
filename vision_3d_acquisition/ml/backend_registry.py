from __future__ import annotations

from typing import Any

from vision_3d_acquisition.ml.backends import ClassifierBackend, make_backend


BACKEND_REGISTRY: dict[str, dict[str, Any]] = {
    "heuristic_ruleset": {
        "id": "heuristic_ruleset",
        "backend_name": "heuristic_ruleset",
        "backend_version": "builtin-v1",
        "interpretability_level": "high",
        "training_speed": "instant",
        "inference_speed": "fast",
        "feature_requirements": {"requires_mm_calibrated_features": True},
        "calibration_requirements": {"requires_mm_calibration": True},
        "operational_compatibility": {"pipeline_families": ["25d"]},
        "description": "Rule-based compatibility baseline and fallback.",
    },
    "logistic_regression": {
        "id": "logistic_regression",
        "backend_name": "sklearn",
        "backend_version": "1.x",
        "interpretability_level": "high",
        "training_speed": "fast",
        "inference_speed": "fast",
        "feature_requirements": {"requires_mm_calibrated_features": True},
        "calibration_requirements": {"requires_mm_calibration": True},
        "operational_compatibility": {"pipeline_families": ["25d"]},
        "description": "Fast and interpretable tabular baseline.",
    },
    "random_forest": {
        "id": "random_forest",
        "backend_name": "sklearn",
        "backend_version": "1.x",
        "interpretability_level": "medium",
        "training_speed": "medium",
        "inference_speed": "medium",
        "feature_requirements": {"requires_mm_calibrated_features": True},
        "calibration_requirements": {"requires_mm_calibration": True},
        "operational_compatibility": {"pipeline_families": ["25d"]},
        "description": "Robust tabular classifier with feature importance.",
    },
    "gradient_boosting": {
        "id": "gradient_boosting",
        "backend_name": "sklearn",
        "backend_version": "1.x",
        "interpretability_level": "medium",
        "training_speed": "slow",
        "inference_speed": "medium",
        "feature_requirements": {"requires_mm_calibrated_features": True},
        "calibration_requirements": {"requires_mm_calibration": True},
        "operational_compatibility": {"pipeline_families": ["25d"]},
        "description": "High-accuracy tabular learner with slower training.",
    },
    "svm": {
        "id": "svm",
        "backend_name": "sklearn",
        "backend_version": "1.x",
        "interpretability_level": "low",
        "training_speed": "medium",
        "inference_speed": "medium",
        "feature_requirements": {"requires_mm_calibrated_features": True},
        "calibration_requirements": {"requires_mm_calibration": True},
        "operational_compatibility": {"pipeline_families": ["25d"]},
        "description": "Useful for compact feature spaces; lower interpretability.",
    },
    "pytorch": {
        "id": "pytorch",
        "backend_name": "pytorch",
        "backend_version": "placeholder",
        "interpretability_level": "variable",
        "training_speed": "variable",
        "inference_speed": "variable",
        "feature_requirements": {"status": "placeholder"},
        "calibration_requirements": {"status": "placeholder"},
        "operational_compatibility": {"pipeline_families": []},
        "description": "Future backend placeholder.",
        "enabled": False,
    },
    "onnx": {
        "id": "onnx",
        "backend_name": "onnx",
        "backend_version": "placeholder",
        "interpretability_level": "variable",
        "training_speed": "variable",
        "inference_speed": "variable",
        "feature_requirements": {"status": "placeholder"},
        "calibration_requirements": {"status": "placeholder"},
        "operational_compatibility": {"pipeline_families": []},
        "description": "Future backend placeholder.",
        "enabled": False,
    },
    "tensorflow": {
        "id": "tensorflow",
        "backend_name": "tensorflow",
        "backend_version": "placeholder",
        "interpretability_level": "variable",
        "training_speed": "variable",
        "inference_speed": "variable",
        "feature_requirements": {"status": "placeholder"},
        "calibration_requirements": {"status": "placeholder"},
        "operational_compatibility": {"pipeline_families": []},
        "description": "Future backend placeholder.",
        "enabled": False,
    },
}


def get_available_backends(*, include_placeholders: bool = True) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for key in sorted(BACKEND_REGISTRY.keys()):
        row = dict(BACKEND_REGISTRY[key])
        if not include_placeholders and row.get("enabled") is False:
            continue
        out.append(row)
    return out


def load_backend(backend_id: str) -> ClassifierBackend:
    normalized = backend_id.strip().lower()
    if normalized == "heuristic_ruleset":
        normalized = "heuristic"
    if normalized not in {"heuristic", "logistic_regression", "random_forest", "gradient_boosting", "svm"}:
        raise ValueError(f"unsupported backend: {backend_id}")
    return make_backend(normalized)
