from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from vision_3d_acquisition.api.settings import ApiSettings
from vision_3d_acquisition.pipelines.registry import list_pipelines
from vision_3d_acquisition.processes.models import ProcessBinding, ProcessPurpose, RecipeVersion


class ProcessBindingService:
    def __init__(self, settings: ApiSettings) -> None:
        self.settings = settings
        self.root = settings.data_dir / "processes"
        self.bindings_path = self.root / "bindings.json"
        self.recipes_dir = self.root / "recipes"
        self.root.mkdir(parents=True, exist_ok=True)
        self.recipes_dir.mkdir(parents=True, exist_ok=True)

    def list_bindings(self) -> list[dict[str, Any]]:
        items = sorted(self._load_bindings(), key=lambda item: item.updated_at, reverse=True)
        return [item.model_dump(mode="json") for item in items]

    def create_binding(
        self,
        *,
        name: str,
        source_id: str,
        modality: str,
        purpose: ProcessPurpose,
        pipeline_id: str,
        active_recipe_version_id: str,
        calibration_profile_id: str | None,
        enabled: bool,
    ) -> dict[str, Any]:
        now = datetime.now(UTC)
        binding = ProcessBinding(
            id=f"binding_{_new_id()}",
            name=name,
            source_id=source_id,
            modality=modality,
            purpose=purpose,
            pipeline_id=pipeline_id,
            active_recipe_version_id=active_recipe_version_id,
            calibration_profile_id=calibration_profile_id,
            enabled=enabled,
            created_at=now,
            updated_at=now,
        )
        bindings = self._load_bindings()
        self._validate_binding(binding, bindings=bindings, updating_id=None)
        bindings.append(binding)
        self._write_bindings(bindings)
        return binding.model_dump(mode="json")

    def update_binding(self, binding_id: str, updates: dict[str, Any]) -> dict[str, Any]:
        bindings = self._load_bindings()
        index = next((i for i, item in enumerate(bindings) if item.id == binding_id), -1)
        if index < 0:
            raise KeyError(f"Unknown process binding: {binding_id}")
        current = bindings[index]
        payload = current.model_dump(mode="json")
        payload.update(updates)
        payload["id"] = current.id
        payload["created_at"] = current.created_at
        payload["updated_at"] = datetime.now(UTC)
        updated = ProcessBinding.model_validate(payload)
        self._validate_binding(updated, bindings=bindings, updating_id=binding_id)
        bindings[index] = updated
        self._write_bindings(bindings)
        return updated.model_dump(mode="json")

    def activate_recipe(self, binding_id: str, recipe_version_id: str) -> dict[str, Any]:
        return self.update_binding(binding_id, {"active_recipe_version_id": recipe_version_id})

    def resolve_process_binding(self, source_id: str, modality: str, purpose: ProcessPurpose) -> dict[str, Any] | None:
        matches = [
            item
            for item in self._load_bindings()
            if item.enabled and item.source_id == source_id and item.modality == modality and item.purpose == purpose
        ]
        if not matches:
            return None
        if len(matches) > 1:
            raise ValueError(
                f"Ambiguous enabled process bindings for source_id={source_id}, modality={modality}, purpose={purpose}."
            )
        return matches[0].model_dump(mode="json")

    def recipe_version(self, recipe_version_id: str) -> RecipeVersion:
        path = self.recipes_dir / f"{recipe_version_id}.json"
        if not path.is_file():
            raise KeyError(f"Unknown recipe version: {recipe_version_id}")
        return RecipeVersion.model_validate(json.loads(path.read_text(encoding="utf-8")))

    def _validate_binding(self, binding: ProcessBinding, *, bindings: list[ProcessBinding], updating_id: str | None) -> None:
        pipeline = next((item for item in list_pipelines() if str(item.get("id") or "") == binding.pipeline_id), None)
        if pipeline is None:
            raise ValueError(f"Unknown pipeline: {binding.pipeline_id}")
        supported_modalities = [str(item) for item in (pipeline.get("supported_modalities") or [])]
        if binding.modality not in supported_modalities:
            raise ValueError(
                f"Pipeline '{binding.pipeline_id}' does not support modality '{binding.modality}'. "
                f"Supported: {', '.join(supported_modalities) or 'none'}"
            )

        recipe = self.recipe_version(binding.active_recipe_version_id)
        if not self._recipe_matches_pipeline(recipe, binding.pipeline_id):
            raise ValueError(
                f"Recipe version '{binding.active_recipe_version_id}' does not belong to pipeline '{binding.pipeline_id}'."
            )

        if not binding.enabled:
            return
        collisions = [
            item
            for item in bindings
            if item.enabled
            and item.source_id == binding.source_id
            and item.modality == binding.modality
            and item.purpose == binding.purpose
            and item.id != updating_id
        ]
        if collisions:
            raise ValueError(
                "Ambiguous enabled bindings are not allowed for the same source_id + modality + purpose."
            )

    @staticmethod
    def _recipe_matches_pipeline(recipe: RecipeVersion, pipeline_id: str) -> bool:
        snapshot = recipe.pipeline_snapshot if isinstance(recipe.pipeline_snapshot, dict) else {}
        snapshot_pipeline_id = str(snapshot.get("pipeline_id") or "")
        if snapshot_pipeline_id:
            return snapshot_pipeline_id == pipeline_id
        template_id = str(recipe.template_id or "")
        if template_id == "mining_steel_ball_classification_2d_reflectance_mvp":
            return pipeline_id == "mining_steel_ball_classification_2d"
        return False

    def _load_bindings(self) -> list[ProcessBinding]:
        payload = _read_json(self.bindings_path)
        rows = payload.get("bindings") if isinstance(payload.get("bindings"), list) else []
        result: list[ProcessBinding] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            result.append(ProcessBinding.model_validate(row))
        return result

    def _write_bindings(self, items: list[ProcessBinding]) -> None:
        self.bindings_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "bindings": [item.model_dump(mode="json") for item in items],
            "updated_at": datetime.now(UTC).isoformat(),
        }
        self.bindings_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _new_id() -> str:
    return datetime.now(UTC).strftime("%Y%m%d%H%M%S%f")
