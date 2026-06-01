from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


SEVERITY_BOUNDS = {
    "LOW": (0.0, 0.3),
    "MEDIUM": (0.3, 0.7),
    "HIGH": (0.7, 1.0),
}


@dataclass
class FeatureWarning:
    feature: str
    severity: str
    message: str
    group: str
    ux_visibility: list[str] = field(default_factory=lambda: ["studio", "classifier_studio"])

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature": self.feature,
            "severity": self.severity,
            "message": self.message,
            "group": self.group,
            "ux_visibility": self.ux_visibility,
        }


@dataclass
class FeatureGroupSummary:
    group: str
    status: str
    severity_score: float
    readiness: str
    top_warnings: list[dict[str, Any]] = field(default_factory=list)
    key_evidence_features: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "group": self.group,
            "status": self.status,
            "severity_score": self.severity_score,
            "readiness": self.readiness,
            "top_warnings": self.top_warnings,
            "key_evidence_features": self.key_evidence_features,
        }


@dataclass
class FeatureReadinessSummary:
    overall_readiness: str
    group_readiness: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "overall_readiness": self.overall_readiness,
            "group_readiness": self.group_readiness,
        }


def severity_from_score(score: float) -> str:
    s = float(max(0.0, min(1.0, score)))
    if s < SEVERITY_BOUNDS["MEDIUM"][0]:
        return "LOW"
    if s < SEVERITY_BOUNDS["HIGH"][0]:
        return "MEDIUM"
    return "HIGH"


def build_object_feature_ux_summary(obj: dict[str, Any]) -> dict[str, Any]:
    fp = obj.get("footprint_geometry") if isinstance(obj.get("footprint_geometry"), dict) else {}
    sg = obj.get("surface_geometry") if isinstance(obj.get("surface_geometry"), dict) else {}
    sc = obj.get("sphere_consistency") if isinstance(obj.get("sphere_consistency"), dict) else {}
    dm = obj.get("damage_metrics") if isinstance(obj.get("damage_metrics"), dict) else {}
    scores = {
        "footprint_geometry": _clamp01(_num(fp.get("radial_cv")) / 0.35),
        "surface_geometry": _clamp01(_num(sg.get("sphere_fit_rmse_mm")) / 8.0),
        "sphere_consistency": _clamp01(_num(sc.get("radial_height_rmse_mm")) / 8.0),
        "damage_metrics": _clamp01((_num(dm.get("flat_region_ratio")) / 0.65 + _num(dm.get("surface_discontinuity_score")) / 1.5) / 2.0),
    }
    warnings: list[FeatureWarning] = []
    if _num(sc.get("radial_height_rmse_mm")) > 8.0:
        warnings.append(FeatureWarning("radial_height_rmse_mm", "HIGH", "Strong sphere-consistency mismatch detected", "sphere_consistency"))
    if _num(dm.get("surface_discontinuity_score")) > 0.7:
        warnings.append(FeatureWarning("surface_discontinuity_score", "HIGH", "Strong surface discontinuity detected", "damage_metrics"))
    if _num(fp.get("radial_cv")) > 0.18:
        warnings.append(FeatureWarning("radial_cv", "MEDIUM", "Boundary irregularity detected", "footprint_geometry"))
    if _num(sg.get("sphere_fit_rmse_mm")) > 4.0:
        warnings.append(FeatureWarning("sphere_fit_rmse_mm", "MEDIUM", "Sphere fit residual elevated", "surface_geometry"))

    group_readiness: dict[str, str] = {}
    group_summaries: list[dict[str, Any]] = []
    for group, score in scores.items():
        sev = severity_from_score(score)
        status = "GOOD" if sev == "LOW" else ("WARNING" if sev == "MEDIUM" else "CRITICAL")
        readiness = "PRODUCTION_READY" if score < 0.30 else ("GOOD" if score < 0.65 else "EXPERIMENTAL")
        group_readiness[group] = readiness
        feature_name, feature_value = _key_evidence_for_group(group, fp, sg, sc, dm)
        group_warns = [w.to_dict() for w in warnings if w.group == group][:3]
        group_summaries.append(
            FeatureGroupSummary(
                group=group,
                status=status,
                severity_score=round(score, 4),
                readiness=readiness,
                top_warnings=group_warns,
                key_evidence_features=[{"feature": feature_name, "value": feature_value, "severity": sev}],
            ).to_dict()
        )
    overall = "PRODUCTION_READY"
    if any(v == "EXPERIMENTAL" for v in group_readiness.values()):
        overall = "EXPERIMENTAL"
    elif any(v == "GOOD" for v in group_readiness.values()):
        overall = "GOOD"
    readiness = FeatureReadinessSummary(overall_readiness=overall, group_readiness=group_readiness).to_dict()
    return {
        "feature_group_summaries": group_summaries,
        "feature_warnings": [w.to_dict() for w in warnings],
        "feature_readiness": readiness,
    }


def filter_for_operations(payload: dict[str, Any]) -> dict[str, Any]:
    groups = payload.get("feature_group_summaries") if isinstance(payload.get("feature_group_summaries"), list) else []
    summary = {}
    for row in groups:
        if not isinstance(row, dict):
            continue
        summary[_ops_label(str(row.get("group") or ""))] = row.get("status")
    warnings = []
    for w in payload.get("feature_warnings") or []:
        if isinstance(w, dict) and "operations" in (w.get("ux_visibility") or []):
            warnings.append({"severity": w.get("severity"), "message": w.get("message")})
    return {"operational_summary": summary, "warnings": warnings}


def filter_for_studio(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "feature_group_summaries": payload.get("feature_group_summaries") or [],
        "feature_warnings": [w for w in (payload.get("feature_warnings") or []) if isinstance(w, dict) and "studio" in (w.get("ux_visibility") or [])],
        "feature_readiness": payload.get("feature_readiness") or {},
    }


def filter_for_classifier_studio(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "feature_group_summaries": payload.get("feature_group_summaries") or [],
        "feature_warnings": [w for w in (payload.get("feature_warnings") or []) if isinstance(w, dict) and "classifier_studio" in (w.get("ux_visibility") or [])],
        "feature_readiness": payload.get("feature_readiness") or {},
    }


def _ops_label(group: str) -> str:
    mapping = {
        "surface_geometry": "surface_coherence",
        "damage_metrics": "damage_evidence",
        "sphere_consistency": "sphere_consistency",
        "footprint_geometry": "footprint_shape",
    }
    return mapping.get(group, group)


def _key_evidence_for_group(group: str, fp: dict[str, Any], sg: dict[str, Any], sc: dict[str, Any], dm: dict[str, Any]) -> tuple[str, float]:
    if group == "footprint_geometry":
        return ("radial_cv", _num(fp.get("radial_cv")))
    if group == "surface_geometry":
        return ("sphere_fit_rmse_mm", _num(sg.get("sphere_fit_rmse_mm")))
    if group == "sphere_consistency":
        return ("radial_height_rmse_mm", _num(sc.get("radial_height_rmse_mm")))
    return ("surface_discontinuity_score", _num(dm.get("surface_discontinuity_score")))


def _num(v: Any) -> float:
    try:
        return float(v)
    except Exception:
        return 0.0


def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, float(v)))
