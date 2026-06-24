from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_RULE_PARAMS: dict[str, float] = {
    "good_min_height_mm": 8.0,
    "good_max_height_mm": 95.0,
    "good_min_sphericity": 0.8,
    "good_max_eccentricity": 0.65,
    "good_min_flatness": 0.15,
    "good_max_edge_roughness": 8.0,
    "deformed_min_eccentricity": 0.85,
    "deformed_max_flatness": -0.2,
    "deformed_min_edge_roughness": 12.0,
    "ball_scrap_min_sphericity": 0.45,
    "scrap_max_sphericity": 0.30,
    "scrap_max_height_mm": 6.0,
    "scrap_max_p95_height_mm": 4.0,
    "scrap_max_volume_proxy_mm3": 4000.0,
    "scrap_min_flatness": -0.35,
    "fallback_scrap_max_sphericity": 0.30,
    "fallback_ball_scrap_max_sphericity": 0.75,
}


@dataclass
class RulePrediction:
    superclass: str
    rule_path: str
    confidence_proxy: float


@dataclass
class ResolvedRuleSet:
    classifier_engine: str
    rule_set_id: str
    rule_set_version: str
    rule_set_path: str | None
    rule_set_source: str
    params: dict[str, float]
    warnings: list[str]
    metadata: dict[str, Any]


def _safe_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _metric(row: dict[str, Any], key: str) -> float | None:
    direct = _safe_float(row.get(key))
    if direct is not None:
        return direct
    return _safe_float(row.get(f"diag_{key}"))


def _margin_high_is_good(value: float | None, threshold: float) -> float:
    if value is None:
        return 0.0
    return max(0.0, min(1.0, (value - threshold) / max(1e-9, abs(threshold) + 1.0)))


def _margin_high_is_bad(value: float | None, threshold: float) -> float:
    if value is None:
        return 0.0
    return max(0.0, min(1.0, (threshold - value) / max(1e-9, abs(threshold) + 1.0)))


def _margin_range(value: float | None, low: float, high: float) -> float:
    if value is None or value < low or value > high:
        return 0.0
    center = (low + high) / 2.0
    half = max(1e-9, (high - low) / 2.0)
    return max(0.0, 1.0 - abs(value - center) / half)


def predict_superclass_from_rules(row: dict[str, Any], params: dict[str, float]) -> RulePrediction:
    sph3d = _metric(row, "feature_sphericity_3d")
    max_h = _metric(row, "max_height_mm")
    p95_h = _metric(row, "p95_height_mm")
    volume = _metric(row, "feature_volume_proxy_mm3")
    flat = _metric(row, "feature_flatness")
    ecc = _metric(row, "feature_eccentricity")
    rough = _metric(row, "feature_edge_roughness")
    border_touch = _metric(row, "border_touch_ratio")
    invalid_ratio = _metric(row, "invalid_pixel_ratio")

    if sph3d is not None and sph3d <= params["scrap_max_sphericity"]:
        return RulePrediction("SCRAP_METAL", "matched_scrap_low_sphericity", _margin_high_is_bad(sph3d, params["scrap_max_sphericity"]))
    if max_h is not None and max_h <= params["scrap_max_height_mm"]:
        return RulePrediction("SCRAP_METAL", "matched_scrap_low_height", _margin_high_is_bad(max_h, params["scrap_max_height_mm"]))
    if p95_h is not None and p95_h <= params["scrap_max_p95_height_mm"]:
        return RulePrediction("SCRAP_METAL", "matched_scrap_low_p95_height", _margin_high_is_bad(p95_h, params["scrap_max_p95_height_mm"]))
    if volume is not None and volume <= params["scrap_max_volume_proxy_mm3"]:
        return RulePrediction("SCRAP_METAL", "matched_scrap_low_volume", _margin_high_is_bad(volume, params["scrap_max_volume_proxy_mm3"]))
    if flat is not None and flat <= params["scrap_min_flatness"]:
        return RulePrediction("SCRAP_METAL", "matched_scrap_low_flatness", _margin_high_is_bad(flat, params["scrap_min_flatness"]))
    if border_touch is not None and border_touch >= 0.30 and invalid_ratio is not None and invalid_ratio >= 0.35:
        return RulePrediction("SCRAP_METAL", "matched_scrap_fragmented_border", 0.7)

    good_checks = [
        max_h is not None and params["good_min_height_mm"] <= max_h <= params["good_max_height_mm"],
        sph3d is not None and sph3d >= params["good_min_sphericity"],
        ecc is not None and ecc <= params["good_max_eccentricity"],
        flat is not None and flat >= params["good_min_flatness"],
        rough is not None and rough <= params["good_max_edge_roughness"],
    ]
    if all(good_checks):
        margins = [
            _margin_range(max_h, params["good_min_height_mm"], params["good_max_height_mm"]),
            _margin_high_is_good(sph3d, params["good_min_sphericity"]),
            _margin_high_is_bad(ecc, params["good_max_eccentricity"]),
            _margin_high_is_good(flat, params["good_min_flatness"]),
            _margin_high_is_bad(rough, params["good_max_edge_roughness"]),
        ]
        return RulePrediction("BALL_GOOD", "matched_good_ball", min(margins))

    ball_like = sph3d is not None and sph3d >= params["ball_scrap_min_sphericity"]
    deformed = (
        (ecc is not None and ecc >= params["deformed_min_eccentricity"])
        or (flat is not None and flat <= params["deformed_max_flatness"])
        or (rough is not None and rough >= params["deformed_min_edge_roughness"])
    )
    if ball_like and deformed:
        return RulePrediction("BALL_SCRAP", "matched_ball_scrap_deformation", 0.65)

    if sph3d is not None and sph3d < params["fallback_scrap_max_sphericity"]:
        return RulePrediction("SCRAP_METAL", "fallback_scrap", _margin_high_is_bad(sph3d, params["fallback_scrap_max_sphericity"]))
    if sph3d is not None and sph3d < params["fallback_ball_scrap_max_sphericity"]:
        return RulePrediction("BALL_SCRAP", "fallback_ball_scrap", _margin_high_is_bad(sph3d, params["fallback_ball_scrap_max_sphericity"]))
    return RulePrediction("BALL_SCRAP", "fallback_ball_scrap_default", 0.50)


def load_classifier_rule_config(path: str | Path) -> dict[str, Any]:
    cfg_path = Path(path)
    if not cfg_path.is_file():
        raise ValueError(f"classifier rule config not found: {cfg_path}")
    try:
        payload = json.loads(cfg_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"failed to parse classifier rule config: {cfg_path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("classifier rule config must be a JSON object")
    params_raw = payload.get("params")
    if params_raw is not None and not isinstance(params_raw, dict):
        raise ValueError("classifier rule config field 'params' must be an object")
    params = dict(DEFAULT_RULE_PARAMS)
    unknown: list[str] = []
    for key, value in (params_raw or {}).items():
        if key not in DEFAULT_RULE_PARAMS:
            unknown.append(str(key))
            continue
        parsed = _safe_float(value)
        if parsed is None:
            continue
        params[key] = parsed
    return {
        "classifier_id": payload.get("classifier_id") or "mining_steel_ball_classification_25d_rules",
        "version": payload.get("version") or "external",
        "description": payload.get("description"),
        "rule_order": payload.get("rule_order") or ["SCRAP_METAL", "BALL_GOOD", "BALL_SCRAP", "FALLBACK"],
        "metadata": payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {},
        "params": params,
        "unknown_params": sorted(unknown),
        "path": str(cfg_path),
    }


def resolve_classifier_rule_set(
    *,
    runtime_override_path: str | Path | None = None,
    pipeline_config_path: str | Path | None = None,
    env_default_path: str | Path | None = None,
) -> ResolvedRuleSet:
    warnings: list[str] = []
    env_path = env_default_path if env_default_path is not None else os.environ.get("SENSOR_STUDIO_DEFAULT_RULE_SET")
    candidate_sources: list[tuple[str, str | Path | None]] = [
        ("runtime_override", runtime_override_path),
        ("pipeline_config", pipeline_config_path),
        ("env_default", env_path),
    ]
    for source, candidate in candidate_sources:
        if not candidate:
            continue
        cfg = load_classifier_rule_config(candidate)
        unknown = [str(item) for item in (cfg.get("unknown_params") or []) if str(item)]
        if unknown:
            warnings.append(f"unknown params ignored: {', '.join(sorted(unknown))}")
        cfg_path = str(cfg.get("path") or candidate)
        return ResolvedRuleSet(
            classifier_engine=str(cfg.get("classifier_id") or "mining_steel_ball_classification_25d_rules"),
            rule_set_id=Path(cfg_path).stem if cfg_path else "external",
            rule_set_version=str(cfg.get("version") or "external"),
            rule_set_path=cfg_path or None,
            rule_set_source=source,
            params=dict(cfg.get("params") or DEFAULT_RULE_PARAMS),
            warnings=warnings,
            metadata=dict(cfg.get("metadata") or {}),
        )
    return ResolvedRuleSet(
        classifier_engine="mining_steel_ball_classification_25d_rules",
        rule_set_id="builtin_default",
        rule_set_version="builtin",
        rule_set_path=None,
        rule_set_source="builtin_default",
        params=dict(DEFAULT_RULE_PARAMS),
        warnings=[],
        metadata={},
    )


def list_available_rule_sets(config_dir: str | Path = "configs/classifiers") -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    root = Path(config_dir)
    if not root.is_dir():
        return out
    for path in sorted(root.glob("*.json")):
        try:
            cfg = load_classifier_rule_config(path)
        except ValueError:
            continue
        meta = cfg.get("metadata") if isinstance(cfg.get("metadata"), dict) else {}
        out.append(
            {
                "rule_set_id": Path(path).stem,
                "classifier_id": cfg.get("classifier_id"),
                "version": cfg.get("version"),
                "description": cfg.get("description"),
                "created_at": meta.get("created_at"),
                "dataset_id": meta.get("dataset_id"),
                "ml_set_id": meta.get("ml_set_id"),
                "optimized_metric": meta.get("optimized_metric"),
                "validation_macro_f1": meta.get("validation_macro_f1"),
                "source_type": str(meta.get("source_type") or "tuned"),
                "path": str(path),
            }
        )
    return out
