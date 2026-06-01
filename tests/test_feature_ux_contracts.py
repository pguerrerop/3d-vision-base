from __future__ import annotations

from vision_3d_acquisition.ml.features.ux_contracts import (
    SEVERITY_BOUNDS,
    build_object_feature_ux_summary,
    filter_for_classifier_studio,
    filter_for_operations,
    filter_for_studio,
    severity_from_score,
)


def test_severity_mapping_is_deterministic() -> None:
    assert SEVERITY_BOUNDS["LOW"] == (0.0, 0.3)
    assert severity_from_score(0.1) == "LOW"
    assert severity_from_score(0.3) == "MEDIUM"
    assert severity_from_score(0.7) == "HIGH"
    assert severity_from_score(1.5) == "HIGH"
    assert severity_from_score(-3.0) == "LOW"


def test_group_summaries_and_warning_extraction() -> None:
    obj = {
        "footprint_geometry": {"radial_cv": 0.21},
        "surface_geometry": {"sphere_fit_rmse_mm": 5.2},
        "sphere_consistency": {"radial_height_rmse_mm": 9.4},
        "damage_metrics": {"flat_region_ratio": 0.82, "surface_discontinuity_score": 1.2},
    }
    summary = build_object_feature_ux_summary(obj)
    groups = summary.get("feature_group_summaries") or []
    warnings = summary.get("feature_warnings") or []
    readiness = summary.get("feature_readiness") or {}
    assert len(groups) == 4
    assert len(warnings) >= 3
    assert readiness.get("overall_readiness") in {"GOOD", "EXPERIMENTAL", "PRODUCTION_READY"}
    assert isinstance(readiness.get("group_readiness"), dict)


def test_visibility_filters_are_audience_aware() -> None:
    payload = {
        "feature_group_summaries": [{"group": "surface_geometry", "status": "GOOD"}],
        "feature_warnings": [
            {"severity": "HIGH", "message": "ops", "ux_visibility": ["operations"]},
            {"severity": "LOW", "message": "studio", "ux_visibility": ["studio"]},
            {"severity": "LOW", "message": "classifier", "ux_visibility": ["classifier_studio"]},
        ],
        "feature_readiness": {"overall_readiness": "GOOD"},
    }
    ops = filter_for_operations(payload)
    studio = filter_for_studio(payload)
    cls = filter_for_classifier_studio(payload)
    assert "operational_summary" in ops
    assert len(ops["warnings"]) == 1
    assert len(studio["feature_warnings"]) == 1
    assert len(cls["feature_warnings"]) == 1
