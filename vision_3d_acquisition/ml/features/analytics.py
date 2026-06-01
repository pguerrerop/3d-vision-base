from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

from vision_3d_acquisition.ml.features.dataset import FeatureDataset
from vision_3d_acquisition.ml.features.registry import FeatureRegistry


@dataclass
class FeatureAnalyticsReports:
    quality_report: dict[str, Any]
    stability_report: dict[str, Any]
    correlation_report: dict[str, Any]
    readiness_report: dict[str, Any]
    drift_report: dict[str, Any]
    ux_summary: dict[str, Any]


def run_feature_analytics(
    dataset: FeatureDataset,
    registry: FeatureRegistry,
    *,
    baseline_snapshot: dict[str, Any] | None = None,
) -> FeatureAnalyticsReports:
    quality = compute_feature_quality_report(dataset, registry)
    correlation = compute_feature_correlation_report(dataset, registry)
    stability = compute_feature_stability_report(dataset, registry)
    readiness = compute_feature_readiness_report(
        dataset=dataset,
        registry=registry,
        quality_report=quality,
        stability_report=stability,
        correlation_report=correlation,
    )
    drift = compute_feature_drift_report(dataset, baseline_snapshot=baseline_snapshot)
    ux = build_feature_ux_summary(
        dataset=dataset,
        registry=registry,
        readiness_report=readiness,
        quality_report=quality,
    )
    return FeatureAnalyticsReports(
        quality_report=quality,
        stability_report=stability,
        correlation_report=correlation,
        readiness_report=readiness,
        drift_report=drift,
        ux_summary=ux,
    )


def compute_feature_quality_report(dataset: FeatureDataset, registry: FeatureRegistry) -> dict[str, Any]:
    names = dataset.get_feature_names()
    feature_rows: dict[str, dict[str, Any]] = {}
    low_variance: list[str] = []
    for name in names:
        values = [sample.feature_vector.get(name) for sample in dataset.samples]
        missing = [v for v in values if v is None]
        nums = [float(v) for v in values if v is not None and math.isfinite(float(v))]
        invalid = [v for v in values if v is not None and (not _is_finite_float(v))]
        miss_ratio = len(missing) / max(1, len(values))
        invalid_ratio = len(invalid) / max(1, len(values))
        mu = _mean(nums) if nums else 0.0
        sd = _stdev(nums) if nums else 0.0
        med = _median(nums) if nums else 0.0
        mad = _median([abs(v - med) for v in nums]) if nums else 0.0
        outliers = 0
        extreme = 0
        if nums and sd > 1e-12:
            for v in nums:
                z = abs((v - mu) / sd)
                if z > 3.0:
                    outliers += 1
                if z > 6.0:
                    extreme += 1
        robust_outliers = 0
        if nums and mad > 1e-12:
            for v in nums:
                rz = abs((v - med) / (1.4826 * mad))
                if rz > 3.5:
                    robust_outliers += 1
        var = sd * sd
        entropy_proxy = _entropy_proxy(nums)
        if var < 1e-9:
            low_variance.append(name)
        feature_rows[name] = {
            "missing_ratio": miss_ratio,
            "invalid_ratio": invalid_ratio,
            "mean": mu if nums else None,
            "stddev": sd if nums else None,
            "variance": var if nums else None,
            "outlier_ratio": outliers / max(1, len(nums)) if nums else 0.0,
            "extreme_value_count": extreme,
            "robust_outlier_ratio": robust_outliers / max(1, len(nums)) if nums else 0.0,
            "entropy_proxy": entropy_proxy,
            "expected_range": list(registry.features.get(name).expected_range) if registry.features.get(name) and registry.features.get(name).expected_range else None,
            "recommended_warning_threshold": (registry.features.get(name).recommended_warning_threshold if registry.features.get(name) else None),
            "recommended_critical_threshold": (registry.features.get(name).recommended_critical_threshold if registry.features.get(name) else None),
        }
    group_summary = _group_aggregate(dataset, names, feature_rows)
    warnings = [f"low_variance:{name}" for name in low_variance]
    return {
        "feature_schema_version": dataset.feature_schema.feature_schema_version,
        "sample_count": len(dataset.samples),
        "features": feature_rows,
        "groups": group_summary,
        "warnings": warnings,
    }


def compute_feature_stability_report(dataset: FeatureDataset, registry: FeatureRegistry) -> dict[str, Any]:
    names = dataset.get_feature_names()
    by_seed_group: dict[str, list[Any]] = {}
    for sample in dataset.samples:
        key = str(sample.metadata.get("synthetic_object_type") or sample.label or "unknown")
        by_seed_group.setdefault(key, []).append(sample)
    feature_stability: dict[str, dict[str, Any]] = {}
    for name in names:
        per_group_var = []
        per_group_mean = []
        for group_name, samples in by_seed_group.items():
            vals = [sample.feature_vector.get(name) for sample in samples if sample.feature_vector.get(name) is not None]
            nums = [float(v) for v in vals]
            if not nums:
                continue
            per_group_var.append(_stdev(nums) ** 2)
            per_group_mean.append(abs(_mean(nums)))
        perturbation_variance = _mean(per_group_var) if per_group_var else 0.0
        signal = _mean(per_group_mean) if per_group_mean else 0.0
        noise_sensitivity = perturbation_variance / max(signal * signal, 1e-9)
        stability_score = _clamp01(1.0 - min(1.0, noise_sensitivity))
        feature_stability[name] = {
            "perturbation_variance": perturbation_variance,
            "noise_sensitivity_score": noise_sensitivity,
            "feature_stability_score": stability_score,
        }
    ranking = sorted(
        [{"feature": name, **row} for name, row in feature_stability.items()],
        key=lambda item: float(item.get("feature_stability_score") or 0.0),
        reverse=True,
    )
    group_stability: dict[str, dict[str, Any]] = {}
    for group, features in registry.groups.items():
        scores = [float(feature_stability.get(name, {}).get("feature_stability_score") or 0.0) for name in features]
        group_stability[group] = {
            "group_stability_score": _mean(scores) if scores else 0.0,
            "feature_count": len(features),
        }
    return {
        "feature_schema_version": dataset.feature_schema.feature_schema_version,
        "features": feature_stability,
        "ranking": ranking,
        "groups": group_stability,
    }


def compute_feature_correlation_report(dataset: FeatureDataset, registry: FeatureRegistry) -> dict[str, Any]:
    names = dataset.get_feature_names()
    cols = {name: [sample.feature_vector.get(name) for sample in dataset.samples] for name in names}
    matrix: dict[str, dict[str, float]] = {name: {} for name in names}
    high_corr_pairs: list[dict[str, Any]] = []
    for i, a in enumerate(names):
        for b in names[i:]:
            corr = _pearson(cols[a], cols[b])
            matrix[a][b] = corr
            if a != b:
                matrix[b][a] = corr
                if abs(corr) >= 0.95:
                    high_corr_pairs.append({"feature_a": a, "feature_b": b, "correlation": corr})
    group_warnings = []
    for pair in high_corr_pairs:
        ga = registry.features.get(pair["feature_a"]).group if registry.features.get(pair["feature_a"]) else "unknown"
        gb = registry.features.get(pair["feature_b"]).group if registry.features.get(pair["feature_b"]) else "unknown"
        if ga == gb:
            group_warnings.append(f"redundant_within_group:{ga}:{pair['feature_a']}:{pair['feature_b']}")
    return {
        "feature_schema_version": dataset.feature_schema.feature_schema_version,
        "correlation_matrix": matrix,
        "high_correlation_pairs": high_corr_pairs,
        "warnings": sorted(set(group_warnings)),
    }


def compute_feature_readiness_report(
    *,
    dataset: FeatureDataset,
    registry: FeatureRegistry,
    quality_report: dict[str, Any],
    stability_report: dict[str, Any],
    correlation_report: dict[str, Any],
) -> dict[str, Any]:
    redundant = set()
    for pair in correlation_report.get("high_correlation_pairs") or []:
        redundant.add(str(pair.get("feature_b")))
    rows: dict[str, dict[str, Any]] = {}
    for name in dataset.get_feature_names():
        q = (quality_report.get("features") or {}).get(name) or {}
        s = (stability_report.get("features") or {}).get(name) or {}
        missing = float(q.get("missing_ratio") or 0.0)
        invalid = float(q.get("invalid_ratio") or 0.0)
        outlier = float(q.get("outlier_ratio") or 0.0)
        stability = float(s.get("feature_stability_score") or 0.0)
        variance = float(q.get("variance") or 0.0)
        variance_score = _clamp01(min(1.0, variance / 1e-2))
        redundancy_penalty = 0.35 if name in redundant else 0.0
        interpretability = 1.0 if registry.features.get(name) and registry.features[name].description else 0.7
        readiness = _clamp01(
            (1.0 - missing) * 0.25
            + (1.0 - invalid) * 0.20
            + (1.0 - min(1.0, outlier)) * 0.10
            + stability * 0.20
            + variance_score * 0.10
            + interpretability * 0.15
            - redundancy_penalty
        )
        level = "EXPERIMENTAL"
        if readiness >= 0.8:
            level = "PRODUCTION_READY"
        elif readiness >= 0.6:
            level = "GOOD"
        rows[name] = {
            "readiness_score": readiness,
            "readiness_level": level,
            "components": {
                "missingness": missing,
                "invalidity": invalid,
                "outlier_ratio": outlier,
                "stability": stability,
                "variance_score": variance_score,
                "redundancy_penalty": redundancy_penalty,
                "interpretability": interpretability,
            },
        }
    group_rows = {}
    for group, features in registry.groups.items():
        vals = [float((rows.get(name) or {}).get("readiness_score") or 0.0) for name in features]
        group_rows[group] = {
            "readiness_score": _mean(vals) if vals else 0.0,
            "readiness_level": _readiness_level(_mean(vals) if vals else 0.0),
            "feature_count": len(features),
        }
    return {"features": rows, "groups": group_rows}


def compute_distribution_by_object_type(dataset: FeatureDataset, registry: FeatureRegistry) -> dict[str, Any]:
    out: dict[str, Any] = {}
    bucket: dict[str, list[Any]] = {}
    for sample in dataset.samples:
        key = str(sample.metadata.get("synthetic_object_type") or sample.label or "unknown")
        bucket.setdefault(key, []).append(sample)
    for object_type, samples in bucket.items():
        feature_stats = {}
        for name in dataset.get_feature_names():
            vals = [sample.feature_vector.get(name) for sample in samples if sample.feature_vector.get(name) is not None]
            nums = [float(v) for v in vals]
            if not nums:
                continue
            nums_sorted = sorted(nums)
            feature_stats[name] = {
                "mean": _mean(nums),
                "stddev": _stdev(nums),
                "p05": _percentile(nums_sorted, 5),
                "p50": _percentile(nums_sorted, 50),
                "p95": _percentile(nums_sorted, 95),
            }
        out[object_type] = feature_stats
    return out


def compute_feature_drift_report(dataset: FeatureDataset, *, baseline_snapshot: dict[str, Any] | None = None) -> dict[str, Any]:
    current = dataset.compute_statistics().get("features") or {}
    if not baseline_snapshot:
        return {"status": "baseline_only", "baseline_snapshot": {"features": current}, "drift": {}}
    base = (baseline_snapshot.get("features") if isinstance(baseline_snapshot, dict) else {}) or {}
    drift_rows = {}
    for name, stats in current.items():
        b = base.get(name) if isinstance(base.get(name), dict) else None
        if not b:
            continue
        mean_shift = _safe_sub(stats.get("mean"), b.get("mean"))
        std_shift = _safe_sub(stats.get("stddev"), b.get("stddev"))
        drift_rows[name] = {
            "mean_shift": mean_shift,
            "stddev_shift": std_shift,
            "drift_score": abs(mean_shift) + abs(std_shift),
        }
    return {"status": "compared", "baseline_snapshot": baseline_snapshot, "drift": drift_rows}


def build_feature_ux_summary(
    *,
    dataset: FeatureDataset,
    registry: FeatureRegistry,
    readiness_report: dict[str, Any],
    quality_report: dict[str, Any],
) -> dict[str, Any]:
    group_cards = []
    for group, names in registry.groups.items():
        rvals = [float((readiness_report.get("features") or {}).get(name, {}).get("readiness_score") or 0.0) for name in names]
        qvals = [float((quality_report.get("features") or {}).get(name, {}).get("missing_ratio") or 0.0) for name in names]
        readiness = _mean(rvals) if rvals else 0.0
        health = "GOOD"
        if readiness < 0.45:
            health = "HIGH_RISK"
        elif readiness < 0.7:
            health = "MEDIUM"
        group_cards.append(
            {
                "group": group,
                "operations_visibility": True,
                "studio_visibility": True,
                "classifier_studio_visibility": True,
                "semantic_summary": f"{group.replace('_', ' ').title()}: {health}",
                "readiness_score": readiness,
                "missing_ratio_mean": _mean(qvals) if qvals else 0.0,
            }
        )
    return {
        "operations": {"group_health_cards": group_cards},
        "studio": {"group_health_cards": group_cards},
        "classifier_studio": {
            "group_health_cards": group_cards,
            "feature_readiness": readiness_report.get("features") or {},
            "feature_quality": quality_report.get("features") or {},
        },
    }


def export_feature_analytics_reports(reports: FeatureAnalyticsReports, output_dir: Path) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "feature_quality_report": str(output_dir / "feature_quality_report.json"),
        "feature_stability_report": str(output_dir / "feature_stability_report.json"),
        "feature_correlation_report": str(output_dir / "feature_correlation_report.json"),
        "feature_readiness_report": str(output_dir / "feature_readiness_report.json"),
        "feature_drift_report": str(output_dir / "feature_drift_report.json"),
        "feature_ux_summary": str(output_dir / "feature_ux_summary.json"),
    }
    (output_dir / "feature_quality_report.json").write_text(json.dumps(reports.quality_report, indent=2), encoding="utf-8")
    (output_dir / "feature_stability_report.json").write_text(json.dumps(reports.stability_report, indent=2), encoding="utf-8")
    (output_dir / "feature_correlation_report.json").write_text(json.dumps(reports.correlation_report, indent=2), encoding="utf-8")
    (output_dir / "feature_readiness_report.json").write_text(json.dumps(reports.readiness_report, indent=2), encoding="utf-8")
    (output_dir / "feature_drift_report.json").write_text(json.dumps(reports.drift_report, indent=2), encoding="utf-8")
    (output_dir / "feature_ux_summary.json").write_text(json.dumps(reports.ux_summary, indent=2), encoding="utf-8")
    return paths


def _group_aggregate(dataset: FeatureDataset, names: list[str], feature_rows: dict[str, dict[str, Any]]) -> dict[str, Any]:
    out = {}
    groups = dataset.get_feature_groups()
    for group, feats in groups.items():
        variances = [float((feature_rows.get(name) or {}).get("variance") or 0.0) for name in feats]
        missing = [float((feature_rows.get(name) or {}).get("missing_ratio") or 0.0) for name in feats]
        out[group] = {
            "feature_count": len(feats),
            "group_variance": _mean(variances) if variances else 0.0,
            "group_missing_ratio": _mean(missing) if missing else 0.0,
        }
    return out


def _readiness_level(score: float) -> str:
    if score >= 0.8:
        return "PRODUCTION_READY"
    if score >= 0.6:
        return "GOOD"
    return "EXPERIMENTAL"


def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, float(v)))


def _is_finite_float(v: Any) -> bool:
    try:
        return math.isfinite(float(v))
    except Exception:
        return False


def _mean(values: list[float]) -> float:
    return float(sum(values) / max(1, len(values)))


def _stdev(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mu = _mean(values)
    return float(math.sqrt(sum((x - mu) ** 2 for x in values) / (len(values) - 1)))


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    n = len(s)
    mid = n // 2
    if n % 2 == 1:
        return float(s[mid])
    return float((s[mid - 1] + s[mid]) / 2.0)


def _percentile(sorted_values: list[float], p: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    idx = (len(sorted_values) - 1) * (p / 100.0)
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return float(sorted_values[lo])
    w = idx - lo
    return float(sorted_values[lo] * (1.0 - w) + sorted_values[hi] * w)


def _entropy_proxy(values: list[float]) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    bins = 12
    lo, hi = s[0], s[-1]
    if hi <= lo:
        return 0.0
    step = (hi - lo) / bins
    counts = [0 for _ in range(bins)]
    for v in values:
        idx = int((v - lo) / step)
        if idx >= bins:
            idx = bins - 1
        if idx < 0:
            idx = 0
        counts[idx] += 1
    total = float(sum(counts))
    ent = 0.0
    for c in counts:
        if c <= 0:
            continue
        p = c / total
        ent -= p * math.log(max(p, 1e-12))
    return ent / math.log(float(bins))


def _pearson(a_raw: list[Any], b_raw: list[Any]) -> float:
    pairs = []
    for a, b in zip(a_raw, b_raw):
        if a is None or b is None:
            continue
        if not _is_finite_float(a) or not _is_finite_float(b):
            continue
        pairs.append((float(a), float(b)))
    if len(pairs) < 2:
        return 0.0
    ax = [p[0] for p in pairs]
    bx = [p[1] for p in pairs]
    am = _mean(ax)
    bm = _mean(bx)
    num = sum((x - am) * (y - bm) for x, y in pairs)
    den_a = math.sqrt(sum((x - am) ** 2 for x in ax))
    den_b = math.sqrt(sum((y - bm) ** 2 for y in bx))
    den = den_a * den_b
    if den <= 1e-12:
        return 0.0
    return float(num / den)


def _safe_sub(a: Any, b: Any) -> float:
    try:
        if a is None or b is None:
            return 0.0
        return float(a) - float(b)
    except Exception:
        return 0.0
