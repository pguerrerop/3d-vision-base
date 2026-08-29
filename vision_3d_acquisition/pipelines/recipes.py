from __future__ import annotations

from datetime import UTC, datetime
import json
import re
from pathlib import Path
from typing import Any, Mapping

from vision_3d_acquisition.api.settings import ApiSettings
from vision_3d_acquisition.pipelines.processing_units import (
    PROCESSING_UNIT_REGISTRY_VERSION,
    processing_unit_contract_fingerprint,
    recipe_parameters_by_unit,
    stage_params_from_recipe_parameters,
    validate_processing_unit_contracts,
)
from vision_3d_acquisition.pipelines.registry import get_pipeline, list_processing_unit_definitions


def _slugify(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower()).strip("_")
    return normalized or "recipe"


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _json_copy(value: Mapping[str, Any] | None) -> dict[str, Any]:
    return json.loads(json.dumps(dict(value or {})))


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


class RecipeService:
    def __init__(self, settings: ApiSettings) -> None:
        self.settings = settings
        self.root = settings.data_dir / "recipes"
        self.root.mkdir(parents=True, exist_ok=True)

    def list_recipes(self, pipeline_id: str, *, include_archived: bool = False) -> list[dict[str, Any]]:
        pipeline_dir = self.root / pipeline_id
        if not pipeline_dir.is_dir():
            return []
        items: list[dict[str, Any]] = []
        for path in pipeline_dir.glob("*/recipe.json"):
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not include_archived and bool(payload.get("archived")):
                continue
            enriched = self._with_compatibility(dict(payload))
            items.append(enriched)
        items.sort(key=lambda item: (str(item.get("updated_at") or ""), str(item.get("name") or "")), reverse=True)
        return items

    def get_recipe(self, pipeline_id: str, recipe_id: str, *, version: int | None = None) -> dict[str, Any]:
        path = self._recipe_version_path(pipeline_id, recipe_id, version)
        if not path.is_file():
            raise KeyError(f"Unknown recipe: {recipe_id}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        return self._with_compatibility(payload)

    def create_recipe(
        self,
        pipeline_id: str,
        *,
        name: str,
        description: str | None = None,
        recipe_id: str | None = None,
        tags: list[str] | None = None,
        notes: str | None = None,
        parameters_by_unit: Mapping[str, Any] | None = None,
        stage_params: Mapping[str, Any] | None = None,
        unit_enabled_state: Mapping[str, Any] | None = None,
        artifact_policy: Mapping[str, Any] | None = None,
        calibration_snapshot_reference: Mapping[str, Any] | None = None,
        provenance: Mapping[str, Any] | None = None,
        parent_recipe_id: str | None = None,
    ) -> dict[str, Any]:
        normalized_id = recipe_id or f"recipe_{_slugify(name)}"
        if self._recipe_current_path(pipeline_id, normalized_id).exists():
            raise ValueError(f"Recipe already exists: {normalized_id}")
        payload = self._build_recipe_payload(
            pipeline_id=pipeline_id,
            recipe_id=normalized_id,
            version=1,
            name=name,
            description=description,
            tags=tags,
            notes=notes,
            parameters_by_unit=parameters_by_unit,
            stage_params=stage_params,
            unit_enabled_state=unit_enabled_state,
            artifact_policy=artifact_policy,
            calibration_snapshot_reference=calibration_snapshot_reference,
            provenance=provenance,
            parent_recipe_id=parent_recipe_id,
            existing_created_at=None,
        )
        self._write_recipe(payload)
        return self._with_compatibility(payload)

    def update_recipe(
        self,
        pipeline_id: str,
        recipe_id: str,
        *,
        name: str | None = None,
        description: str | None = None,
        tags: list[str] | None = None,
        notes: str | None = None,
        parameters_by_unit: Mapping[str, Any] | None = None,
        stage_params: Mapping[str, Any] | None = None,
        unit_enabled_state: Mapping[str, Any] | None = None,
        artifact_policy: Mapping[str, Any] | None = None,
        calibration_snapshot_reference: Mapping[str, Any] | None = None,
        provenance: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        current = self.get_recipe(pipeline_id, recipe_id)
        payload = self._build_recipe_payload(
            pipeline_id=pipeline_id,
            recipe_id=recipe_id,
            version=int(current.get("version") or 0) + 1,
            name=name or str(current.get("name") or recipe_id),
            description=description if description is not None else current.get("description"),
            tags=tags if tags is not None else list(current.get("tags") or []),
            notes=notes if notes is not None else current.get("notes"),
            parameters_by_unit=parameters_by_unit if parameters_by_unit is not None else current.get("parameters_by_unit"),
            stage_params=stage_params if stage_params is not None else current.get("stage_params"),
            unit_enabled_state=unit_enabled_state if unit_enabled_state is not None else current.get("unit_enabled_state"),
            artifact_policy=artifact_policy if artifact_policy is not None else current.get("artifact_policy"),
            calibration_snapshot_reference=(
                calibration_snapshot_reference
                if calibration_snapshot_reference is not None
                else current.get("calibration_snapshot_reference")
            ),
            provenance=provenance if provenance is not None else current.get("provenance"),
            parent_recipe_id=str(current.get("parent_recipe_id") or "") or None,
            existing_created_at=str(current.get("created_at") or "") or None,
        )
        self._write_recipe(payload)
        return self._with_compatibility(payload)

    def clone_recipe(
        self,
        pipeline_id: str,
        recipe_id: str,
        *,
        new_name: str | None = None,
        new_recipe_id: str | None = None,
    ) -> dict[str, Any]:
        current = self.get_recipe(pipeline_id, recipe_id)
        clone_name = new_name or f"{current.get('name') or recipe_id} copy"
        return self.create_recipe(
            pipeline_id,
            name=clone_name,
            description=current.get("description"),
            recipe_id=new_recipe_id,
            tags=list(current.get("tags") or []),
            notes=current.get("notes"),
            parameters_by_unit=current.get("parameters_by_unit"),
            stage_params=current.get("stage_params"),
            unit_enabled_state=current.get("unit_enabled_state"),
            artifact_policy=current.get("artifact_policy"),
            calibration_snapshot_reference=current.get("calibration_snapshot_reference"),
            provenance={
                **(current.get("provenance") if isinstance(current.get("provenance"), Mapping) else {}),
                "cloned_from_recipe_id": recipe_id,
            },
            parent_recipe_id=recipe_id,
        )

    def archive_recipe(self, pipeline_id: str, recipe_id: str) -> dict[str, Any]:
        current = self.get_recipe(pipeline_id, recipe_id)
        current["archived"] = True
        current["updated_at"] = _now_iso()
        self._write_recipe(current, write_version=False)
        return self._with_compatibility(current)

    def validate_recipe(
        self,
        pipeline_id: str,
        *,
        recipe_id: str | None = None,
        parameters_by_unit: Mapping[str, Any] | None = None,
        stage_params: Mapping[str, Any] | None = None,
        stored_contract_fingerprint: str | None = None,
        archived: bool | None = None,
    ) -> dict[str, Any]:
        pipeline = get_pipeline(pipeline_id)
        if pipeline is None:
            return {
                "compatibility": "incompatible",
                "errors": [f"Unknown pipeline: {pipeline_id}"],
                "warnings": [],
            }
        units = list_processing_unit_definitions(pipeline_id)
        contract_check = validate_processing_unit_contracts(units)
        errors = list(contract_check.get("errors") or [])
        warnings = list(contract_check.get("warnings") or [])
        current_fingerprint = processing_unit_contract_fingerprint(units)
        if stored_contract_fingerprint and stored_contract_fingerprint != current_fingerprint:
            warnings.append(
                "Contract fingerprint mismatch: recipe was saved against a different processing-unit contract."
            )
        if archived is True:
            warnings.append("Archived recipe selected.")
        if recipe_id and not str(recipe_id).strip():
            errors.append("recipe_id must not be empty when provided.")

        normalized_parameters = self._normalize_parameters(units, parameters_by_unit, stage_params)
        unit_by_id = {str(unit.get("id") or ""): unit for unit in units if isinstance(unit, Mapping)}
        for unit_id, params in normalized_parameters.items():
            unit = unit_by_id.get(unit_id)
            if unit is None:
                warnings.append(f"Unknown unit id in recipe: {unit_id}")
                continue
            param_defs = {
                str(item.get("id") or ""): item
                for item in unit.get("parameters") or []
                if isinstance(item, Mapping) and item.get("id")
            }
            for param_id, value in params.items():
                param = param_defs.get(param_id)
                if param is None:
                    warnings.append(f"Unknown or legacy parameter on {unit_id}: {param_id}")
                    continue
                param_type = str(param.get("type") or "")
                if value is None and bool(param.get("nullable")):
                    continue
                if param_type in {"number", "integer"} and not _is_number(value):
                    errors.append(f"{unit_id}.{param_id} expects a numeric value.")
                    continue
                if param_type == "integer" and _is_number(value) and int(value) != value:
                    errors.append(f"{unit_id}.{param_id} expects an integer value.")
                if param_type == "boolean" and not isinstance(value, bool):
                    errors.append(f"{unit_id}.{param_id} expects a boolean value.")
                if param_type == "string" and not isinstance(value, str):
                    errors.append(f"{unit_id}.{param_id} expects a string value.")
                options = param.get("options")
                if isinstance(options, list) and value is not None and str(value) not in {str(item) for item in options}:
                    errors.append(f"{unit_id}.{param_id} must be one of {options}.")
                minimum = param.get("min")
                maximum = param.get("max")
                if _is_number(value) and _is_number(minimum) and float(value) < float(minimum):
                    errors.append(f"{unit_id}.{param_id} must be >= {minimum}.")
                if _is_number(value) and _is_number(maximum) and float(value) > float(maximum):
                    errors.append(f"{unit_id}.{param_id} must be <= {maximum}.")

        compatibility = "incompatible" if errors else "compatible_with_warnings" if warnings else "compatible"
        return {
            "pipeline_id": pipeline_id,
            "recipe_id": recipe_id,
            "compatibility": compatibility,
            "errors": errors,
            "warnings": warnings,
            "normalized_parameters_by_unit": normalized_parameters,
        }

    def recipe_for_run(self, pipeline_id: str, recipe_id: str | None, version: int | None = None) -> dict[str, Any] | None:
        if not recipe_id:
            return None
        return self.get_recipe(pipeline_id, recipe_id, version=version)

    def stage_params_for_recipe(self, pipeline_id: str, recipe_id: str, *, version: int | None = None) -> dict[str, Any]:
        recipe = self.get_recipe(pipeline_id, recipe_id, version=version)
        stage_params = recipe.get("stage_params")
        return _json_copy(stage_params if isinstance(stage_params, Mapping) else {})

    def _build_recipe_payload(
        self,
        *,
        pipeline_id: str,
        recipe_id: str,
        version: int,
        name: str,
        description: str | None,
        tags: list[str] | None,
        notes: str | None,
        parameters_by_unit: Mapping[str, Any] | None,
        stage_params: Mapping[str, Any] | None,
        unit_enabled_state: Mapping[str, Any] | None,
        artifact_policy: Mapping[str, Any] | None,
        calibration_snapshot_reference: Mapping[str, Any] | None,
        provenance: Mapping[str, Any] | None,
        parent_recipe_id: str | None,
        existing_created_at: str | None,
    ) -> dict[str, Any]:
        pipeline = get_pipeline(pipeline_id)
        if pipeline is None:
            raise KeyError(f"Unknown pipeline: {pipeline_id}")
        units = list_processing_unit_definitions(pipeline_id)
        validation = self.validate_recipe(
            pipeline_id,
            recipe_id=recipe_id,
            parameters_by_unit=parameters_by_unit,
            stage_params=stage_params,
        )
        if validation["compatibility"] == "incompatible":
            raise ValueError("; ".join(validation["errors"]) or "Recipe is incompatible.")
        normalized_parameters = validation["normalized_parameters_by_unit"]
        normalized_stage_params = (
            _json_copy(stage_params)
            if isinstance(stage_params, Mapping)
            else stage_params_from_recipe_parameters(units, normalized_parameters)
        )
        now = _now_iso()
        return {
            "recipe_id": recipe_id,
            "name": name,
            "description": description,
            "pipeline_id": pipeline_id,
            "pipeline_family": pipeline.get("pipeline_family"),
            "created_at": existing_created_at or now,
            "updated_at": now,
            "version": version,
            "parent_recipe_id": parent_recipe_id,
            "tags": list(tags or []),
            "notes": notes,
            "processing_unit_contract_version": PROCESSING_UNIT_REGISTRY_VERSION,
            "processing_unit_contract_fingerprint": processing_unit_contract_fingerprint(units),
            "parameters_by_unit": normalized_parameters,
            "stage_params": normalized_stage_params,
            "unit_enabled_state": dict(unit_enabled_state or {}),
            "artifact_policy": dict(artifact_policy or {"diagnostic_level": "engineering", "persist_intermediates": True}),
            "calibration_snapshot_reference": dict(calibration_snapshot_reference or {}),
            "provenance": dict(provenance or {}),
            "archived": False,
        }

    def _normalize_parameters(
        self,
        units: list[dict[str, Any]],
        parameters_by_unit: Mapping[str, Any] | None,
        stage_params: Mapping[str, Any] | None,
    ) -> dict[str, dict[str, Any]]:
        if isinstance(parameters_by_unit, Mapping):
            normalized: dict[str, dict[str, Any]] = {}
            for unit_id, values in parameters_by_unit.items():
                if isinstance(values, Mapping):
                    normalized[str(unit_id)] = dict(values)
            return normalized
        return recipe_parameters_by_unit(units, stage_params)

    def _with_compatibility(self, payload: dict[str, Any]) -> dict[str, Any]:
        validation = self.validate_recipe(
            str(payload.get("pipeline_id") or ""),
            recipe_id=str(payload.get("recipe_id") or ""),
            parameters_by_unit=payload.get("parameters_by_unit") if isinstance(payload.get("parameters_by_unit"), Mapping) else None,
            stage_params=payload.get("stage_params") if isinstance(payload.get("stage_params"), Mapping) else None,
            stored_contract_fingerprint=str(payload.get("processing_unit_contract_fingerprint") or "") or None,
            archived=bool(payload.get("archived")) if payload.get("archived") is not None else None,
        )
        payload["compatibility"] = validation["compatibility"]
        payload["validation_warnings"] = validation["warnings"]
        payload["validation_errors"] = validation["errors"]
        return payload

    def _recipe_dir(self, pipeline_id: str, recipe_id: str) -> Path:
        return self.root / pipeline_id / recipe_id

    def _recipe_current_path(self, pipeline_id: str, recipe_id: str) -> Path:
        return self._recipe_dir(pipeline_id, recipe_id) / "recipe.json"

    def _recipe_version_path(self, pipeline_id: str, recipe_id: str, version: int | None) -> Path:
        if version is None:
            return self._recipe_current_path(pipeline_id, recipe_id)
        return self._recipe_dir(pipeline_id, recipe_id) / "versions" / f"v{version}.json"

    def _write_recipe(self, payload: Mapping[str, Any], *, write_version: bool = True) -> None:
        pipeline_id = str(payload.get("pipeline_id") or "")
        recipe_id = str(payload.get("recipe_id") or "")
        version = int(payload.get("version") or 1)
        recipe_dir = self._recipe_dir(pipeline_id, recipe_id)
        version_dir = recipe_dir / "versions"
        recipe_dir.mkdir(parents=True, exist_ok=True)
        version_dir.mkdir(parents=True, exist_ok=True)
        current_path = recipe_dir / "recipe.json"
        current_path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True), encoding="utf-8")
        if write_version:
            version_path = version_dir / f"v{version}.json"
            version_path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True), encoding="utf-8")
