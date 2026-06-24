#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from statistics import mean
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from vision_3d_acquisition.apps.ball_inspection_25d import run_ball_inspection_25d_flow  # noqa: E402
from vision_3d_acquisition.synthetic import SyntheticNoiseConfig, create_advanced_synthetic_25d_take  # noqa: E402


OBJECT_TYPES = [
    "good_ball",
    "worn_ellipsoid",
    "truncated_sphere",
    "chipped_sphere",
    "flattened_ball",
    "elongated_scrap",
]


FEATURE_METADATA: dict[str, dict[str, Any]] = {
    "circularity": {
        "group": "footprint_geometry",
        "unit": "ratio",
        "higher_is_worse": False,
        "description": "Isoperimetric contour circularity.",
    },
    "radial_cv": {
        "group": "footprint_geometry",
        "unit": "ratio",
        "higher_is_worse": True,
        "description": "Radial boundary coefficient of variation.",
    },
    "eccentricity": {
        "group": "footprint_geometry",
        "unit": "ratio",
        "higher_is_worse": True,
        "description": "Ellipse eccentricity in XY footprint.",
    },
    "sphere_fit_rmse_mm": {
        "group": "surface_geometry",
        "unit": "mm",
        "higher_is_worse": True,
        "description": "Sphere fit residual RMS over visible surface.",
    },
    "sphere_vs_ellipsoid_gain": {
        "group": "surface_geometry",
        "unit": "ratio",
        "higher_is_worse": True,
        "description": "Ellipsoid-vs-sphere fit improvement ratio.",
    },
    "deformation_score": {
        "group": "surface_geometry",
        "unit": "ratio",
        "higher_is_worse": True,
        "description": "Combined deformation severity proxy.",
    },
    "radial_height_rmse_mm": {
        "group": "sphere_consistency",
        "unit": "mm",
        "higher_is_worse": True,
        "description": "Observed vs spherical-cap radial residual RMS.",
    },
    "surface_completeness_ratio": {
        "group": "sphere_consistency",
        "unit": "ratio",
        "higher_is_worse": False,
        "description": "Visible cap completeness ratio.",
    },
    "volume_deficit_ratio": {
        "group": "sphere_consistency",
        "unit": "ratio",
        "higher_is_worse": True,
        "description": "Observed-vs-expected volume deficit ratio.",
    },
    "surface_roughness_score": {
        "group": "damage_metrics",
        "unit": "ratio",
        "higher_is_worse": True,
        "description": "Normalized local roughness score.",
    },
    "flat_region_ratio": {
        "group": "damage_metrics",
        "unit": "ratio",
        "higher_is_worse": True,
        "description": "Fraction of planar/flat region support.",
    },
    "surface_discontinuity_score": {
        "group": "damage_metrics",
        "unit": "score",
        "higher_is_worse": True,
        "description": "Sharp transition/discontinuity score.",
    },
}

EXPECTED_DOMINANT_BY_TYPE = {
    "good_ball": "none",
    "worn_ellipsoid": "surface_geometry",
    "truncated_sphere": "sphere_consistency",
    "chipped_sphere": "damage_metrics",
    "flattened_ball": "damage_metrics",
    "elongated_scrap": "footprint_geometry",
}


def _grade(value: float | None, *, good_max: float, medium_max: float) -> str:
    if value is None:
        return "UNKNOWN"
    if value <= good_max:
        return "PASS"
    if value <= medium_max:
        return "MEDIUM"
    return "FAIL"


def _family_summary(obj: dict[str, Any]) -> dict[str, str]:
    fp = obj.get("footprint_geometry") if isinstance(obj.get("footprint_geometry"), dict) else {}
    sg = obj.get("surface_geometry") if isinstance(obj.get("surface_geometry"), dict) else {}
    sc = obj.get("sphere_consistency") if isinstance(obj.get("sphere_consistency"), dict) else {}
    dm = obj.get("damage_metrics") if isinstance(obj.get("damage_metrics"), dict) else {}
    return {
        "footprint_geometry": _grade(_as_float(fp.get("radial_cv")), good_max=0.12, medium_max=0.2),
        "surface_geometry": _grade(_as_float(sg.get("sphere_fit_rmse_mm")), good_max=7.0, medium_max=12.0),
        "sphere_consistency": _grade(_as_float(sc.get("radial_height_rmse_mm")), good_max=6.0, medium_max=11.0),
        "damage_metrics": _grade(_as_float(dm.get("flat_region_ratio")), good_max=0.25, medium_max=0.55),
    }


def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, float(v)))


def _severity_scale(value: float | None, *, good_max: float, fail_max: float) -> float:
    if value is None:
        return 0.0
    if value <= good_max:
        return 0.0
    if fail_max <= good_max:
        return 1.0
    return _clamp01((float(value) - good_max) / (fail_max - good_max))


def _inverse_severity_scale(value: float | None, *, good_min: float, fail_min: float) -> float:
    if value is None:
        return 0.0
    if value >= good_min:
        return 0.0
    if fail_min >= good_min:
        return 1.0
    return _clamp01((good_min - float(value)) / (good_min - fail_min))


def _family_scores(metrics: dict[str, float | None]) -> dict[str, float]:
    return {
        "footprint_geometry": _clamp01(mean([
            _severity_scale(metrics.get("radial_cv"), good_max=0.10, fail_max=0.35),
            _inverse_severity_scale(metrics.get("circularity"), good_min=0.90, fail_min=0.65),
            _severity_scale(metrics.get("eccentricity"), good_max=0.20, fail_max=0.90),
        ])),
        "surface_geometry": _clamp01(mean([
            _severity_scale(metrics.get("sphere_fit_rmse_mm"), good_max=1.0, fail_max=8.0),
            _severity_scale(metrics.get("deformation_score"), good_max=0.15, fail_max=0.75),
            _severity_scale(metrics.get("sphere_vs_ellipsoid_gain"), good_max=0.05, fail_max=0.45),
        ])),
        "sphere_consistency": _clamp01(mean([
            _severity_scale(metrics.get("radial_height_rmse_mm"), good_max=0.8, fail_max=8.0),
            _inverse_severity_scale(metrics.get("surface_completeness_ratio"), good_min=0.98, fail_min=0.65),
            _severity_scale(metrics.get("volume_deficit_ratio"), good_max=0.20, fail_max=0.75),
        ])),
        "damage_metrics": _clamp01(mean([
            _severity_scale(metrics.get("surface_roughness_score"), good_max=0.05, fail_max=0.40),
            _severity_scale(metrics.get("flat_region_ratio"), good_max=0.10, fail_max=0.65),
            _severity_scale(metrics.get("surface_discontinuity_score"), good_max=0.15, fail_max=1.50),
        ])),
    }


def _severity_label(v: float) -> str:
    if v < 0.33:
        return "LOW"
    if v < 0.66:
        return "MEDIUM"
    return "HIGH"


def _expected_dominant_family(row: dict[str, Any]) -> str:
    obj_type = str(row.get("object_type") or "")
    mapped = EXPECTED_DOMINANT_BY_TYPE.get(obj_type)
    if mapped:
        return mapped
    reactions = row.get("expected_metric_family_reactions")
    if not isinstance(reactions, dict):
        return "unknown"
    severity = {"LOW": 0, "PASS": 0, "GOOD": 0, "MEDIUM": 1, "HIGH": 2, "FAIL": 2}
    best = "unknown"
    best_score = -1
    for family, raw in reactions.items():
        score = severity.get(str(raw).upper(), 0)
        if score > best_score:
            best = str(family)
            best_score = score
    return best


def _mean(values: list[float]) -> float:
    return float(sum(values) / max(1, len(values)))


def _safe_stdev(values: list[float]) -> float:
    if len(values) < 2:
        return 1e-9
    mu = _mean(values)
    return float(math.sqrt(sum((v - mu) ** 2 for v in values) / (len(values) - 1)))


def _to_jsonable(row: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in row.items():
        if isinstance(v, (str, int, float, bool)) or v is None:
            out[k] = v
        else:
            out[k] = v
    return out


def _as_float(v: Any) -> float | None:
    try:
        out = float(v)
    except (TypeError, ValueError):
        return None
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Synthetic 25D metric-family validation harness.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Data root")
    parser.add_argument("--session-id", type=str, default="synthetic_25d_geometry_suite", help="Session id")
    parser.add_argument("--samples-per-type", type=int, default=2, help="Samples per object type")
    parser.add_argument("--seed", type=int, default=2201, help="Base seed")
    parser.add_argument("--gaussian-noise", type=float, default=0.35, help="Gaussian height noise stddev in mm")
    parser.add_argument("--missing-fraction", type=float, default=0.01, help="Missing no-return pixel fraction")
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    report_rows: list[dict[str, Any]] = []
    noise = SyntheticNoiseConfig(
        gaussian_height_std_mm=float(args.gaussian_noise),
        missing_region_fraction=float(args.missing_fraction),
        edge_aliasing=True,
    )
    for i, object_type in enumerate(OBJECT_TYPES):
        for j in range(max(1, int(args.samples_per_type))):
            seed = int(args.seed + i * 100 + j)
            take_id, _ = create_advanced_synthetic_25d_take(
                data_dir,
                object_type=object_type,
                session_id=args.session_id,
                seed=seed,
                noise=noise,
            )
            result = run_ball_inspection_25d_flow(data_dir, take_id=take_id).result_payload
            objects = result.get("objects") or []
            primary = max(objects, key=lambda row: float(row.get("footprint_area_mm2") or 0.0), default={})
            fam = _family_summary(primary if isinstance(primary, dict) else {})
            metrics = {
                "circularity": ((primary.get("footprint_geometry") or {}).get("circularity") if isinstance(primary, dict) else None),
                "radial_cv": ((primary.get("footprint_geometry") or {}).get("radial_cv") if isinstance(primary, dict) else None),
                "eccentricity": ((primary.get("footprint_geometry") or {}).get("eccentricity") if isinstance(primary, dict) else None),
                "sphere_fit_rmse_mm": ((primary.get("surface_geometry") or {}).get("sphere_fit_rmse_mm") if isinstance(primary, dict) else None),
                "sphere_vs_ellipsoid_gain": ((primary.get("surface_geometry") or {}).get("sphere_vs_ellipsoid_gain") if isinstance(primary, dict) else None),
                "deformation_score": ((primary.get("surface_geometry") or {}).get("deformation_score") if isinstance(primary, dict) else None),
                "radial_height_rmse_mm": ((primary.get("sphere_consistency") or {}).get("radial_height_rmse_mm") if isinstance(primary, dict) else None),
                "surface_completeness_ratio": ((primary.get("sphere_consistency") or {}).get("surface_completeness_ratio") if isinstance(primary, dict) else None),
                "volume_deficit_ratio": ((primary.get("sphere_consistency") or {}).get("volume_deficit_ratio") if isinstance(primary, dict) else None),
                "surface_roughness_score": ((primary.get("damage_metrics") or {}).get("surface_roughness_score") if isinstance(primary, dict) else None),
                "flat_region_ratio": ((primary.get("damage_metrics") or {}).get("flat_region_ratio") if isinstance(primary, dict) else None),
                "surface_discontinuity_score": ((primary.get("damage_metrics") or {}).get("surface_discontinuity_score") if isinstance(primary, dict) else None),
            }
            family_scores = _family_scores({k: _as_float(v) for k, v in metrics.items()})
            ranking = sorted(family_scores.items(), key=lambda kv: kv[1], reverse=True)
            dominant = ranking[0][0]
            secondary = ranking[1][0]
            md = (result.get("take_metadata") or {}) if isinstance(result.get("take_metadata"), dict) else {}
            expected_failure_modes = md.get("expected_failure_modes") or []
            expected_metric_family_reactions = md.get("expected_metric_family_reactions") or {}
            row = {
                "sample_id": take_id,
                "take_id": take_id,
                "object_type": object_type,
                "synthetic_object_type": object_type,
                "seed": seed,
                "expected_failure_modes": expected_failure_modes,
                "expected_metric_family_reactions": expected_metric_family_reactions,
                "classification": {
                    "class_name": primary.get("class_name") if isinstance(primary, dict) else None,
                    "subclass_label": primary.get("subclass_label") if isinstance(primary, dict) else None,
                    "label": primary.get("label") if isinstance(primary, dict) else None,
                    "superclass": primary.get("superclass") if isinstance(primary, dict) else None,
                    "confidence": primary.get("confidence") if isinstance(primary, dict) else None,
                },
                "family_summary": fam,
                "metrics": metrics,
                "family_scores": family_scores,
                "dominant_failure_family": dominant,
                "secondary_failure_family": secondary,
                "family_severity_ranking": [name for name, _score in ranking],
            }
            report_rows.append(row)
            print(object_type.upper())
            print(f"- take_id: {take_id}")
            print(f"- footprint_geometry: {fam['footprint_geometry']}")
            print(f"- surface_geometry: {fam['surface_geometry']}")
            print(f"- sphere_consistency: {fam['sphere_consistency']}")
            print(f"- damage_metrics: {fam['damage_metrics']}")

    out_dir = data_dir / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    # Baseline calibration from good-ball samples.
    baseline_rows = [r for r in report_rows if str(r.get("object_type")) == "good_ball"]
    baseline_stats: dict[str, dict[str, float]] = {}
    for fname in FEATURE_METADATA:
        vals = [_as_float((r.get("metrics") or {}).get(fname)) for r in baseline_rows]
        nums = [float(v) for v in vals if v is not None]
        if nums:
            baseline_stats[fname] = {"mean": _mean(nums), "std": _safe_stdev(nums)}

    # Per-row z-score deviations from good baseline.
    for row in report_rows:
        metrics = row.get("metrics") if isinstance(row.get("metrics"), dict) else {}
        zdev: dict[str, float] = {}
        for fname, meta in FEATURE_METADATA.items():
            x = _as_float(metrics.get(fname))
            b = baseline_stats.get(fname)
            if x is None or not b:
                continue
            z = (x - b["mean"]) / max(b["std"], 1e-9)
            # normalize direction so positive => worse/deviation
            zdev[fname] = float(z if bool(meta.get("higher_is_worse")) else -z)
        row["zscore_deviation_from_good"] = zdev
        expected_dom = _expected_dominant_family(row)
        observed_dom = str(row.get("dominant_failure_family") or "unknown")
        row["expected_dominant_family"] = expected_dom
        row["dominant_family_match"] = bool(expected_dom == observed_dom or expected_dom == "none")

    # Aggregate by object type.
    by_type: dict[str, list[dict[str, Any]]] = {}
    for row in report_rows:
        by_type.setdefault(str(row.get("object_type") or "unknown"), []).append(row)
    family_sensitivity_matrix: dict[str, dict[str, Any]] = {}
    type_aggregates: dict[str, dict[str, Any]] = {}
    for object_type, rows in by_type.items():
        family_means: dict[str, float] = {}
        for fam_name in ("footprint_geometry", "surface_geometry", "sphere_consistency", "damage_metrics"):
            vals = [float((r.get("family_scores") or {}).get(fam_name) or 0.0) for r in rows]
            family_means[fam_name] = _mean(vals)
        fam_sorted = sorted(family_means.items(), key=lambda kv: kv[1], reverse=True)
        feature_scores: dict[str, float] = {}
        for fname in FEATURE_METADATA:
            zvals = [float((r.get("zscore_deviation_from_good") or {}).get(fname) or 0.0) for r in rows]
            feature_scores[fname] = _mean(zvals)
        feature_sorted = sorted(feature_scores.items(), key=lambda kv: kv[1], reverse=True)
        type_aggregates[object_type] = {
            "average_family_scores": family_means,
            "dominant_failure_family": fam_sorted[0][0],
            "secondary_failure_family": fam_sorted[1][0],
            "family_severity_ranking": [name for name, _score in fam_sorted],
            "strongest_activated_features": [name for name, _s in feature_sorted[:3]],
            "weakest_activated_features": [name for name, _s in sorted(feature_scores.items(), key=lambda kv: kv[1])[:3]],
        }
        family_sensitivity_matrix[object_type] = {
            "footprint_geometry": _severity_label(family_means["footprint_geometry"]),
            "surface_geometry": _severity_label(family_means["surface_geometry"]),
            "sphere_consistency": _severity_label(family_means["sphere_consistency"]),
            "damage_metrics": _severity_label(family_means["damage_metrics"]),
            "footprint_score": family_means["footprint_geometry"],
            "surface_score": family_means["surface_geometry"],
            "consistency_score": family_means["sphere_consistency"],
            "damage_score": family_means["damage_metrics"],
        }

    # Confusion-style dominant-family validation.
    confusion_rows: list[dict[str, Any]] = []
    match_count = 0
    for row in report_rows:
        expected = str(row.get("expected_dominant_family") or "unknown")
        observed = str(row.get("dominant_failure_family") or "unknown")
        match = bool(row.get("dominant_family_match"))
        if match:
            match_count += 1
        confusion_rows.append(
            {
                "sample_id": row.get("sample_id"),
                "synthetic_object_type": row.get("synthetic_object_type"),
                "expected_dominant_family": expected,
                "observed_dominant_family": observed,
                "match": match,
            }
        )

    # Global feature rankings by between-type separation.
    feature_global_scores: dict[str, float] = {}
    for fname in FEATURE_METADATA:
        means = []
        for object_type, rows in by_type.items():
            vals = [_as_float((r.get("metrics") or {}).get(fname)) for r in rows]
            nums = [float(v) for v in vals if v is not None]
            if nums:
                means.append(_mean(nums))
        feature_global_scores[fname] = _safe_stdev(means) if means else 0.0

    json_path = out_dir / "advanced_25d_metric_family_report.json"
    md_path = out_dir / "advanced_25d_metric_family_report.md"
    csv_path = out_dir / "advanced_25d_geometry_validation.csv"
    analysis_path = out_dir / "advanced_25d_geometry_analysis.json"

    json_path.write_text(json.dumps({"session_id": args.session_id, "rows": report_rows}, indent=2), encoding="utf-8")
    analysis_payload = {
        "session_id": args.session_id,
        "feature_schema_version": "v1",
        "feature_groups": {
            "footprint_geometry": ["circularity", "radial_cv", "eccentricity"],
            "surface_geometry": ["sphere_fit_rmse_mm", "sphere_vs_ellipsoid_gain", "deformation_score"],
            "sphere_consistency": ["radial_height_rmse_mm", "surface_completeness_ratio", "volume_deficit_ratio"],
            "damage_metrics": ["surface_roughness_score", "flat_region_ratio", "surface_discontinuity_score"],
        },
        "feature_metadata": FEATURE_METADATA,
        "baseline_calibration": {
            "reference_group": "good_ball",
            "feature_stats": baseline_stats,
        },
        "family_sensitivity_matrix": family_sensitivity_matrix,
        "object_type_aggregates": type_aggregates,
        "dominant_family_confusion": {
            "total_samples": len(report_rows),
            "match_count": match_count,
            "mismatch_count": max(0, len(report_rows) - match_count),
            "rows": confusion_rows,
        },
        "global_feature_ranking": [
            {"feature": name, "between_type_std": score}
            for name, score in sorted(feature_global_scores.items(), key=lambda kv: kv[1], reverse=True)
        ],
        "rows": report_rows,
    }
    analysis_path.write_text(json.dumps(analysis_payload, indent=2), encoding="utf-8")

    csv_columns = [
        "sample_id",
        "synthetic_object_type",
        "seed",
        "expected_failure_modes",
        "expected_metric_family_reactions",
        "predicted_class",
        "predicted_subclass",
        "classification_confidence",
        "circularity",
        "radial_cv",
        "eccentricity",
        "sphere_fit_rmse_mm",
        "sphere_vs_ellipsoid_gain",
        "deformation_score",
        "radial_height_rmse_mm",
        "surface_completeness_ratio",
        "volume_deficit_ratio",
        "surface_roughness_score",
        "flat_region_ratio",
        "surface_discontinuity_score",
        "footprint_family_score",
        "surface_family_score",
        "consistency_family_score",
        "damage_family_score",
        "dominant_failure_family",
        "secondary_failure_family",
        "family_severity_ranking",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_columns)
        writer.writeheader()
        for row in report_rows:
            metrics = row.get("metrics") if isinstance(row.get("metrics"), dict) else {}
            family_scores = row.get("family_scores") if isinstance(row.get("family_scores"), dict) else {}
            cls = row.get("classification") if isinstance(row.get("classification"), dict) else {}
            writer.writerow(
                {
                    "sample_id": row.get("sample_id"),
                    "synthetic_object_type": row.get("synthetic_object_type"),
                    "seed": row.get("seed"),
                    "expected_failure_modes": json.dumps(row.get("expected_failure_modes") or []),
                    "expected_metric_family_reactions": json.dumps(row.get("expected_metric_family_reactions") or {}),
                    "predicted_class": cls.get("class_name"),
                    "predicted_subclass": cls.get("subclass_label"),
                    "classification_confidence": cls.get("confidence"),
                    "circularity": metrics.get("circularity"),
                    "radial_cv": metrics.get("radial_cv"),
                    "eccentricity": metrics.get("eccentricity"),
                    "sphere_fit_rmse_mm": metrics.get("sphere_fit_rmse_mm"),
                    "sphere_vs_ellipsoid_gain": metrics.get("sphere_vs_ellipsoid_gain"),
                    "deformation_score": metrics.get("deformation_score"),
                    "radial_height_rmse_mm": metrics.get("radial_height_rmse_mm"),
                    "surface_completeness_ratio": metrics.get("surface_completeness_ratio"),
                    "volume_deficit_ratio": metrics.get("volume_deficit_ratio"),
                    "surface_roughness_score": metrics.get("surface_roughness_score"),
                    "flat_region_ratio": metrics.get("flat_region_ratio"),
                    "surface_discontinuity_score": metrics.get("surface_discontinuity_score"),
                    "footprint_family_score": family_scores.get("footprint_geometry"),
                    "surface_family_score": family_scores.get("surface_geometry"),
                    "consistency_family_score": family_scores.get("sphere_consistency"),
                    "damage_family_score": family_scores.get("damage_metrics"),
                    "dominant_failure_family": row.get("dominant_failure_family"),
                    "secondary_failure_family": row.get("secondary_failure_family"),
                    "family_severity_ranking": json.dumps(row.get("family_severity_ranking") or []),
                }
            )

    lines = ["# Advanced 25D Metric Family Validation", ""]
    lines.extend(
        [
            "## Family Sensitivity Matrix",
            "",
            "| Object Type | Footprint | Surface | Consistency | Damage |",
            "|---|---|---|---|---|",
        ]
    )
    for object_type in OBJECT_TYPES:
        cell = family_sensitivity_matrix.get(object_type) or {}
        lines.append(
            f"| {object_type} | {cell.get('footprint_geometry', '-')} | {cell.get('surface_geometry', '-')} | {cell.get('sphere_consistency', '-')} | {cell.get('damage_metrics', '-')} |"
        )
    lines.append("")
    for row in report_rows:
        lines.append(f"## {row['object_type']} ({row['take_id']})")
        fam = row["family_summary"]
        lines.append(f"- footprint_geometry: {fam['footprint_geometry']}")
        lines.append(f"- surface_geometry: {fam['surface_geometry']}")
        lines.append(f"- sphere_consistency: {fam['sphere_consistency']}")
        lines.append(f"- damage_metrics: {fam['damage_metrics']}")
        lines.append(f"- dominant_failure_family: {row.get('dominant_failure_family')}")
        lines.append(f"- secondary_failure_family: {row.get('secondary_failure_family')}")
        lines.append(f"- classification: {row['classification']}")
        lines.append("")
    lines.append("## Confusion-Style Dominant Family Validation")
    lines.append("")
    lines.append(
        f"- matches: {match_count}/{len(report_rows)} | mismatches: {max(0, len(report_rows) - match_count)}"
    )
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"json_report: {json_path}")
    print(f"markdown_report: {md_path}")
    print(f"csv_report: {csv_path}")
    print(f"analysis_report: {analysis_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
