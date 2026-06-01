from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from vision_3d_acquisition.ml.features.schemas import FEATURE_COLUMNS, FEATURE_SCHEMA_VERSION, feature_ordering_hash, schema_fingerprint
from vision_3d_acquisition.ml.storage import MLStorage


class MLRuntimeResolver:
    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir
        self.storage = MLStorage(data_dir)

    def resolve_active_deployment(self, *, pipeline_id: str, stage_id: str, pipeline_family: str) -> dict[str, Any]:
        deployments = self.storage.list("deployments")
        candidates = [
            d
            for d in deployments
            if str(d.get("target_pipeline_id") or "") == pipeline_id
            and str(d.get("target_stage_id") or "") == stage_id
            and str(d.get("pipeline_family") or "") == pipeline_family
            and str(d.get("status") or "inactive") == "active"
        ]
        candidates.sort(key=lambda d: str(d.get("activated_at") or d.get("deployed_at") or ""), reverse=True)

        diag = {
            "compatible": False,
            "errors": [],
            "warnings": [],
            "fallback_mode": "heuristic",
        }
        if not candidates:
            diag["warnings"].append("no_active_deployment")
            return {"deployment": None, "diagnostics": diag}

        dep = candidates[0]
        model = self.storage.get("models", str(dep.get("model_id") or ""))
        if not model:
            diag["errors"].append("model_not_found")
            return {"deployment": dep, "diagnostics": diag}

        compat = self.validate_compatibility(deployment=dep, model=model, pipeline_id=pipeline_id, stage_id=stage_id, pipeline_family=pipeline_family)
        return {"deployment": dep, "diagnostics": compat, "model": model}

    def validate_compatibility(self, *, deployment: dict[str, Any], model: dict[str, Any], pipeline_id: str, stage_id: str, pipeline_family: str) -> dict[str, Any]:
        errors: list[str] = []
        warnings: list[str] = []

        if str(deployment.get("target_pipeline_id") or "") != pipeline_id:
            errors.append("pipeline_id_mismatch")
        if str(deployment.get("target_stage_id") or "") != stage_id:
            errors.append("stage_id_mismatch")
        if str(deployment.get("pipeline_family") or "") != pipeline_family:
            errors.append("pipeline_family_mismatch")

        feature_schema = model.get("feature_schema") if isinstance(model.get("feature_schema"), dict) else {}
        model_cols = [str(v) for v in feature_schema.get("columns", [])] if isinstance(feature_schema.get("columns"), list) else []
        if str(feature_schema.get("version") or "") != FEATURE_SCHEMA_VERSION:
            errors.append("feature_schema_version_mismatch")
        if model_cols != FEATURE_COLUMNS:
            errors.append("feature_ordering_mismatch")

        model_artifacts = model.get("artifact_paths") if isinstance(model.get("artifact_paths"), dict) else {}
        model_path = Path(str(model_artifacts.get("model_path") or ""))
        if str(model.get("classifier_family")) != "heuristic" and not model_path.is_file():
            errors.append("model_file_missing")

        backend_meta = model.get("backend") if isinstance(model.get("backend"), dict) else {}
        if not backend_meta:
            warnings.append("backend_metadata_missing")
        else:
            backend_name = str(backend_meta.get("backend_name") or "")
            backend_version = str(backend_meta.get("backend_version") or "")
            if backend_name == "sklearn" and not backend_version.startswith("1."):
                errors.append("unsupported_backend_version")

        label_schema = model.get("label_schema") if isinstance(model.get("label_schema"), dict) else {}
        if not isinstance(label_schema.get("classes"), list):
            warnings.append("label_schema_missing_classes")

        compatible = len(errors) == 0
        return {
            "compatible": compatible,
            "errors": errors,
            "warnings": warnings,
            "fallback_mode": "heuristic" if not compatible else "none",
            "checks": {
                "pipeline": {"pipeline_id": pipeline_id, "stage_id": stage_id, "pipeline_family": pipeline_family},
                "feature_schema": {
                    "expected_version": FEATURE_SCHEMA_VERSION,
                    "model_version": feature_schema.get("version"),
                    "expected_feature_count": len(FEATURE_COLUMNS),
                    "model_feature_count": len(model_cols),
                    "expected_ordering_hash": feature_ordering_hash(FEATURE_COLUMNS),
                    "model_ordering_hash": feature_ordering_hash(model_cols),
                    "expected_schema_fingerprint": schema_fingerprint(FEATURE_SCHEMA_VERSION, FEATURE_COLUMNS),
                    "model_schema_fingerprint": schema_fingerprint(str(feature_schema.get("version") or ""), model_cols),
                },
                "label_schema": label_schema,
                "model_integrity": {"model_path": str(model_path), "exists": model_path.is_file()},
                "calibration": {"status": "assumed_compatible_for_25d_mm_features"},
                "dtype": {"expected": "float", "status": "validated_at_runtime_row_level"},
            },
        }


    def validate_runtime_inputs(self, *, model: dict[str, Any], feature_row: dict[str, Any], object_metrics: dict[str, Any], calibration_available: bool) -> dict[str, Any]:
        errors: list[str] = []
        warnings: list[str] = []
        backend = model.get("backend") if isinstance(model.get("backend"), dict) else {}
        req = backend.get("feature_requirements") if isinstance(backend.get("feature_requirements"), dict) else {}
        required_fields = [str(v) for v in req.get("required_metric_fields", [])] if isinstance(req.get("required_metric_fields"), list) else []
        missing_required: list[str] = []
        for field in required_fields:
            val = feature_row.get(field, object_metrics.get(field))
            if val is None:
                missing_required.append(field)
        if missing_required:
            errors.append("missing_required_metric_fields")

        if bool(req.get("requires_mm_calibration")) and not calibration_available:
            errors.append("mm_calibration_required")

        model_cols = ((model.get("feature_schema") or {}).get("columns") if isinstance(model.get("feature_schema"), dict) else []) or []
        for col in model_cols:
            if col not in feature_row:
                errors.append("feature_missing")
                break
        for col in model_cols:
            if col in feature_row and feature_row[col] is not None and not isinstance(feature_row[col], (int, float)):
                errors.append("feature_dtype_mismatch")
                break

        return {
            "compatible": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "fallback_mode": "heuristic" if errors else "none",
            "required_metric_fields": required_fields,
            "missing_metric_fields": missing_required,
            "calibration_available": calibration_available,
        }
