from __future__ import annotations

import json
import hashlib
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from vision_3d_acquisition.api.settings import ApiSettings
from vision_3d_acquisition.calibration.calibration_storage import load_calibration
from vision_3d_acquisition.contracts.artifacts import normalize_processing_artifacts
from vision_3d_acquisition.processing.status_index import append_process_run_index
from vision_3d_acquisition.processes.executor import PipelineExecutor2D
from vision_3d_acquisition.processes.models import PipelineInstance, PipelineRunRecord, PipelineRuntimeState, PipelineStepConfig, RecipeVersion
from vision_3d_acquisition.processes.templates import get_template, list_templates
from vision_3d_acquisition.runtime_config import read_runtime_config


class ProcessService:
    def __init__(self, settings: ApiSettings) -> None:
        self.settings = settings
        self.root = settings.data_dir / "processes"
        self.instances_dir = self.root / "instances"
        self.recipes_dir = self.root / "recipes"
        self.runs_dir = self.root / "runs"
        self.instances_dir.mkdir(parents=True, exist_ok=True)
        self.recipes_dir.mkdir(parents=True, exist_ok=True)
        self.runs_dir.mkdir(parents=True, exist_ok=True)

    def list_templates(self) -> list[dict[str, Any]]:
        return [template.model_dump(mode="json") for template in list_templates()]

    def list_instances(self) -> list[dict[str, Any]]:
        items: list[PipelineInstance] = []
        for path in self.instances_dir.glob("*.json"):
            payload = json.loads(path.read_text(encoding="utf-8"))
            items.append(PipelineInstance.model_validate(payload))
        items.sort(key=lambda item: item.updated_at, reverse=True)
        return [item.model_dump(mode="json") for item in items]

    def create_instance(self, *, name: str, template_id: str, supported_input_type: str) -> dict[str, Any]:
        template = get_template(template_id)
        if template is None:
            raise KeyError(f"Unknown template: {template_id}")
        now = datetime.now(UTC)
        instance = PipelineInstance(
            id=f"pipeline_{_new_id()}",
            name=name,
            template_id=template_id,
            pipeline_family="2d" if template.category == "image_2d" else "3d" if template.category == "image_3d" else "generic",
            supported_input_type=supported_input_type,
            configured_steps=[
                PipelineStepConfig(
                    step_id=step.step_id,
                    algorithm_key=step.default_algorithm_key,
                    enabled=True,
                    params=dict(template.default_parameters.get(step.step_id, {})),
                    ui_state={"collapsed": False},
                )
                for step in template.steps
            ],
            overrides={},
            runtime_state=PipelineRuntimeState(status="ready"),
            execution_history=[],
            created_at=now,
            updated_at=now,
        )
        self._write_instance(instance)
        return instance.model_dump(mode="json")

    def update_instance(self, instance_id: str, *, configured_steps: list[dict[str, Any]] | None = None, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
        instance = self._load_instance(instance_id)
        if configured_steps is not None:
            instance.configured_steps = [PipelineStepConfig.model_validate(step) for step in configured_steps]
        if overrides is not None:
            instance.overrides = dict(overrides)
        instance.updated_at = datetime.now(UTC)
        self._write_instance(instance)
        return instance.model_dump(mode="json")

    def execute(
        self,
        instance_id: str,
        *,
        image_path: Path,
        recipe_version_id: str | None = None,
        pipeline_id: str | None = None,
        source_id: str | None = None,
        take_id: str | None = None,
        acquisition_group_id: str | None = None,
        calibration_profile_id: str | None = None,
    ) -> dict[str, Any]:
        instance = self._load_instance(instance_id)
        executing_instance = instance
        if recipe_version_id:
            recipe = self._load_recipe(recipe_version_id)
            executing_instance = PipelineInstance.model_validate(recipe.pipeline_snapshot)
            executing_instance.id = instance.id
        recipe_hash = _config_hash_for_instance(executing_instance)
        image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")

        executor = PipelineExecutor2D()
        run_id = f"run_{_new_id()}"
        run_dir = self.runs_dir / instance.id / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        resolved_take_id = take_id or _take_id_from_image_path(self.settings.data_dir, image_path)
        all_artifacts: list[dict[str, Any]] = []
        all_measurements: list[dict[str, Any]] = []
        all_overlays: list[dict[str, Any]] = []
        reports = []
        warnings: list[str] = []
        debug_summary: dict[str, Any] = {}
        state: dict[str, Any] = {}
        runtime_default = read_runtime_config()
        if runtime_default.default_calibration_valid and runtime_default.default_calibration_file:
            try:
                calibration = load_calibration(runtime_default.default_calibration_file)
                if calibration.calibration_type == "camera_2d":
                    state["active_2d_calibration"] = calibration.model_dump(mode="json")
            except Exception:
                pass

        status = "success"
        for step in executing_instance.configured_steps:
            report, result = executor.run_step(step, image=image, existing=state)
            reports.append(report)
            all_artifacts.extend(result.artifacts)
            all_overlays.extend(result.overlays)
            all_measurements.extend(result.measurements)
            warnings.extend(result.warnings)
            if result.debug.get("summary"):
                debug_summary = result.debug["summary"]
            state = dict(result.state)
            if report.status == "failed":
                status = "failed"
                break
            if report.status == "warning" and status != "failed":
                status = "warning"

        persisted = self._persist_intermediate_artifacts(
            run_dir=run_dir,
            source_path=image_path,
            state=state,
            step_reports=reports,
            source_artifact_id="source_rgb_image",
            persisted=True,
            existing_artifacts=all_artifacts,
        )
        all_artifacts.extend(persisted)

        preview_path = self._render_overlay_preview(run_dir=run_dir, source_image=state.get("source_rgb"), overlays=all_overlays)
        if preview_path is not None:
            all_artifacts.append(
                {
                    "artifact_id": "overlay_preview_image",
                    "stage_id": "overlay",
                    "kind": "image",
                    "title": "Overlay preview image",
                    "path": preview_path,
                    "mime_type": "image/png",
                    "preview_available": True,
                    "generated": "explicit",
                    "metadata": {
                        "step_id": "overlay",
                        "algorithm_key": "image_2d.classify.overlay_preview",
                        "source_artifact_id": "source_rgb_image",
                        "coordinate_space": "image_pixel",
                        "image_width": int(state["source_rgb"].shape[1]) if isinstance(state.get("source_rgb"), np.ndarray) else None,
                        "image_height": int(state["source_rgb"].shape[0]) if isinstance(state.get("source_rgb"), np.ndarray) else None,
                        "persisted": True,
                    },
                }
            )

        artifact_payload = normalize_processing_artifacts(
            {"take_id": run_id, "processed_at": datetime.now(UTC).isoformat(), "artifacts": all_artifacts + all_overlays, "objects": []}
        )
        record = PipelineRunRecord(
            run_id=run_id,
            timestamp=datetime.now(UTC),
            status=status,
            pipeline_id=pipeline_id,
            recipe_version_id=recipe_version_id,
            config_snapshot_hash=recipe_hash,
            source_id=source_id,
            take_id=resolved_take_id,
            acquisition_group_id=acquisition_group_id,
            calibration_profile_id=calibration_profile_id,
            parameters={step.step_id: dict(step.params) for step in executing_instance.configured_steps},
            warnings=warnings,
            summary={
                **debug_summary,
                "take_id": resolved_take_id,
                "source_image_path": str(image_path),
                "effective_params": {step.step_id: dict(step.params) for step in executing_instance.configured_steps},
            },
            artifacts=artifact_payload,
            overlays=[item for item in artifact_payload if item.get("kind") == "overlay"],
            measurements=all_measurements,
            step_reports=reports,
        )

        instance.execution_history.insert(0, record)
        instance.execution_history = instance.execution_history[:20]
        instance.runtime_state = PipelineRuntimeState(status=status, last_run_id=run_id, last_error=None if status != "failed" else "Step execution failed")
        instance.updated_at = datetime.now(UTC)
        self._write_instance(instance)
        append_process_run_index(
            self.settings.data_dir,
            take_id=resolved_take_id,
            pipeline_instance_id=instance.id,
            run_id=run_id,
            pipeline_family="2d",
            status=status,
            run_dir=run_dir,
            created_at=record.timestamp.isoformat(),
            pipeline_id=pipeline_id,
            recipe_version_id=recipe_version_id,
            config_snapshot_hash=recipe_hash,
            source_id=source_id,
            acquisition_group_id=acquisition_group_id,
            calibration_profile_id=calibration_profile_id,
        )
        return {"pipeline_instance": instance.model_dump(mode="json"), "run": record.model_dump(mode="json")}

    def preview_segmentation(
        self,
        instance_id: str,
        *,
        image_path: Path,
        stage_id: str,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        instance = self._load_instance(instance_id)
        image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        preview_dir = self.root / "previews" / instance.id / "segmentation_preview"
        preview_dir.mkdir(parents=True, exist_ok=True)

        stage_key = str(stage_id or "segmentation").lower()
        apply_seg_overrides = stage_key == "segmentation" or any(
            key in params for key in (
                "threshold",
                "auto_threshold",
                "invert",
                "morph_op",
                "erode_kernel_size",
                "erode_iterations",
                "dilate_kernel_size",
                "dilate_iterations",
                "close_kernel_size",
                "fill_holes",
                "min_area_px",
                "max_area_px",
                "roi_enabled",
            )
        )
        threshold_value = int(params.get("threshold", 125))
        threshold_value = max(0, min(255, threshold_value))
        auto_threshold = bool(params.get("auto_threshold", False))
        invert = bool(params.get("invert", False))
        morph_op = str(params.get("morph_op", "open_close"))
        erode_kernel_size = max(1, int(params.get("erode_kernel_size", 3)))
        erode_iterations = max(1, int(params.get("erode_iterations", 1)))
        dilate_kernel_size = max(1, int(params.get("dilate_kernel_size", 3)))
        dilate_iterations = max(1, int(params.get("dilate_iterations", 1)))
        close_kernel_size = max(1, int(params.get("close_kernel_size", 5)))
        fill_holes = bool(params.get("fill_holes", True))
        min_area_px = max(0, int(params.get("min_area_px", 120)))
        max_area_px_raw = params.get("max_area_px")
        max_area_px = int(max_area_px_raw) if max_area_px_raw not in {None, ""} else None
        roi_enabled = bool(params.get("roi_enabled", False))

        configured_steps = [step.model_copy(deep=True) for step in instance.configured_steps]
        for step in configured_steps:
            if step.step_id != "threshold":
                if step.step_id == "morphology":
                    if apply_seg_overrides:
                        step.params["morph_op"] = morph_op
                        step.params["operation"] = morph_op
                        step.params["erode_kernel_size"] = erode_kernel_size
                        step.params["erode_iterations"] = erode_iterations
                        step.params["dilate_kernel_size"] = dilate_kernel_size
                        step.params["dilate_iterations"] = dilate_iterations
                        step.params["close_kernel_size"] = close_kernel_size
                        step.params["fill_holes"] = fill_holes
                        step.params["min_area_px"] = min_area_px
                        step.params["max_area_px"] = max_area_px
                        step.params["cleanup_min_area"] = min_area_px
                        step.params["cleanup_max_area"] = max_area_px
                if stage_key == "ellipse_fitting" and step.step_id == "ellipse_fitting":
                    step.params.update(dict(params))
                continue
            if apply_seg_overrides:
                step.params["invert"] = invert
                step.params["roi_enabled"] = roi_enabled
                if auto_threshold:
                    step.params["mode"] = "otsu"
                else:
                    step.params["mode"] = "fixed"
                    step.params["value"] = threshold_value

        executor = PipelineExecutor2D()
        all_artifacts: list[dict[str, Any]] = []
        all_overlays: list[dict[str, Any]] = []
        reports = []
        warnings: list[str] = []
        state: dict[str, Any] = {}
        run_steps = {"input", "crop_roi", "rgb_to_gray", "normalize_lighting", "threshold", "morphology"}
        if stage_key == "ellipse_fitting":
            run_steps.update({"blob_detection", "ellipse_fitting"})

        status = "success"
        for step in configured_steps:
            if step.step_id not in run_steps:
                continue
            report, result = executor.run_step(step, image=image, existing=state)
            reports.append(report)
            all_artifacts.extend(result.artifacts)
            all_overlays.extend(result.overlays)
            warnings.extend(result.warnings)
            state = dict(result.state)
            if stage_key == "segmentation" and step.step_id == "morphology":
                break
            if stage_key == "ellipse_fitting" and step.step_id == "ellipse_fitting":
                break
            if report.status == "failed":
                status = "failed"
                break
            if report.status == "warning" and status != "failed":
                status = "warning"

        persisted = self._persist_intermediate_artifacts(
            run_dir=preview_dir,
            source_path=image_path,
            state=state,
            step_reports=reports,
            source_artifact_id="source_rgb_image",
            persisted=False,
            existing_artifacts=all_artifacts,
        )
        all_artifacts.extend(persisted)
        artifact_payload = normalize_processing_artifacts(
            {
                "take_id": "segmentation_preview",
                "processed_at": datetime.now(UTC).isoformat(),
                "artifacts": all_artifacts + all_overlays,
                "objects": [],
            }
        )
        morph_metrics = state.get("morphology_metrics") if isinstance(state.get("morphology_metrics"), dict) else {}
        threshold_mode = "otsu" if auto_threshold else "fixed"
        return {
            "artifacts": artifact_payload,
            "diagnostics": {
                "status": status,
                "warnings": warnings,
                "step_reports": [item.model_dump(mode="json") for item in reports],
                "preview": True,
                "effective_params": {step.step_id: dict(step.params) for step in configured_steps if step.step_id in run_steps},
                "preview_stage": stage_key,
            },
            "metrics": {
                "component_count": int(morph_metrics.get("components_after", 0)),
                "threshold_coverage": float(morph_metrics.get("threshold_foreground_coverage", 0.0)),
                "cleaned_coverage": float(morph_metrics.get("cleaned_foreground_coverage", 0.0)),
                "removed_components": int(morph_metrics.get("removed_components", 0)),
            },
            "applied_params": {
                "threshold": threshold_value,
                "auto_threshold": auto_threshold,
                "invert": invert,
                "threshold_mode": threshold_mode,
                "morph_op": morph_op,
                "erode_kernel_size": erode_kernel_size,
                "erode_iterations": erode_iterations,
                "dilate_kernel_size": dilate_kernel_size,
                "dilate_iterations": dilate_iterations,
                "close_kernel_size": close_kernel_size,
                "fill_holes": fill_holes,
                "min_area_px": min_area_px,
                "max_area_px": max_area_px,
                "roi_enabled": roi_enabled,
            },
        }

    def _persist_intermediate_artifacts(
        self,
        *,
        run_dir: Path,
        source_path: Path,
        state: dict[str, Any],
        step_reports: list[Any],
        source_artifact_id: str,
        persisted: bool,
        existing_artifacts: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        existing_metadata_by_id = {
            str(item.get("artifact_id")): dict(item.get("metadata") or {})
            for item in (existing_artifacts or [])
            if isinstance(item, dict) and isinstance(item.get("metadata"), dict)
        }
        step_by_id = {item.step_id: item for item in step_reports if hasattr(item, "step_id")}

        def alg(step_id: str, fallback: str) -> str:
            report = step_by_id.get(step_id)
            return str(getattr(report, "algorithm_key", fallback))

        def step_effective_params(step_id: str) -> dict[str, Any]:
            report = step_by_id.get(step_id)
            params = getattr(report, "effective_params", None)
            return dict(params) if isinstance(params, dict) else {}

        emitted: list[dict[str, Any]] = []
        mapping: list[tuple[str, str, str, np.ndarray | None]] = [
            ("source_rgb_image", "input", alg("input", "image_2d.preprocess.input"), state.get("source_rgb")),
            ("roi_image", "crop_roi", alg("crop_roi", "image_2d.preprocess.crop_roi"), state.get("current_rgb") if bool((state.get("roi") or {}).get("enabled")) else None),
            ("grayscale_image", "rgb_to_gray", alg("rgb_to_gray", "image_2d.preprocess.rgb_to_gray"), state.get("gray")),
            ("normalized_grayscale_image", "normalize_lighting", alg("normalize_lighting", "image_2d.preprocess.normalize_lighting"), state.get("normalized_gray")),
            ("threshold_mask", "threshold", alg("threshold", "image_2d.segment.threshold"), state.get("threshold_mask")),
            ("cleaned_mask", "morphology", alg("morphology", "image_2d.segment.morphology"), state.get("cleaned_mask")),
            ("overlay_image", "morphology", alg("morphology", "image_2d.segment.morphology"), state.get("morphology_overlay_image")),
            ("rejected_components_overlay", "morphology", alg("morphology", "image_2d.segment.morphology"), state.get("rejected_components_overlay_image")),
            ("blob_debug_overlay", "blob_detection", alg("blob_detection", "image_2d.detect.blob_contours"), state.get("blob_debug_overlay_image")),
            ("blob_labels", "blob_detection", alg("blob_detection", "image_2d.detect.blob_contours"), state.get("blob_labels_image")),
        ]

        for artifact_id, step_id, algorithm_key, image in mapping:
            if not isinstance(image, np.ndarray):
                continue
            filename = f"{artifact_id}.png"
            path = run_dir / filename
            cv2.imwrite(str(path), image)
            rel = str(path.relative_to(self.settings.data_dir))
            emitted.append(
                {
                    "artifact_id": artifact_id,
                    "stage_id": step_id,
                    "kind": "image",
                    "title": artifact_id.replace("_", " ").title(),
                    "path": rel,
                    "mime_type": "image/png",
                    "preview_available": True,
                    "generated": "explicit",
                    "metadata": {
                        **existing_metadata_by_id.get(artifact_id, {}),
                        "step_id": step_id,
                        "algorithm_key": algorithm_key,
                        "source_artifact_id": source_artifact_id,
                        "image_width": int(image.shape[1]),
                        "image_height": int(image.shape[0]),
                        "coordinate_space": "image_pixel",
                        "semantic_kind": artifact_id,
                        "effective_params": step_effective_params(step_id),
                        "persisted": persisted,
                    },
                }
            )

        if not any(item["artifact_id"] == "source_rgb_image" for item in emitted):
            copied = cv2.imread(str(source_path), cv2.IMREAD_UNCHANGED)
            if copied is not None:
                filename = "source_rgb_image.png"
                out = run_dir / filename
                cv2.imwrite(str(out), copied)
                rel = str(out.relative_to(self.settings.data_dir))
                emitted.append(
                {
                    "artifact_id": "source_rgb_image",
                        "stage_id": "input",
                        "kind": "image",
                        "title": "Source Rgb Image",
                        "path": rel,
                        "mime_type": "image/png",
                        "preview_available": True,
                        "generated": "explicit",
                        "metadata": {
                            "step_id": "input",
                            "algorithm_key": alg("input", "image_2d.preprocess.input"),
                            "source_artifact_id": source_artifact_id,
                            "image_width": int(copied.shape[1]),
                            "image_height": int(copied.shape[0]),
                            "coordinate_space": "image_pixel",
                            "effective_params": step_effective_params("input"),
                            "persisted": persisted,
                        },
                    }
                )

        if isinstance(state.get("morphology_metrics"), dict):
            metrics_path = run_dir / "morphology_metrics.json"
            metrics_path.write_text(json.dumps(state["morphology_metrics"], indent=2), encoding="utf-8")
            emitted.append(
                {
                    "artifact_id": "morphology_metrics",
                    "stage_id": "morphology",
                    "kind": "metric",
                    "title": "Morphology metrics",
                    "path": str(metrics_path.relative_to(self.settings.data_dir)),
                    "mime_type": "application/json",
                    "generated": "explicit",
                    "preview_available": False,
                    "metadata": {
                        "step_id": "morphology",
                        "algorithm_key": alg("morphology", "image_2d.segment.morphology"),
                        "source_artifact_id": source_artifact_id,
                        "semantic_kind": "morphology_metrics",
                        "coordinate_space": "image_pixel",
                        "effective_params": step_effective_params("morphology"),
                        **dict(state["morphology_metrics"]),
                    },
                }
            )
        if isinstance(state.get("morphology_debug_json"), dict):
            payload_path = run_dir / "morphology_debug.json"
            payload_path.write_text(json.dumps(state["morphology_debug_json"], indent=2), encoding="utf-8")
            emitted.append(
                {
                    "artifact_id": "morphology_debug_json",
                    "stage_id": "morphology",
                    "kind": "json",
                    "title": "Morphology debug JSON",
                    "path": str(payload_path.relative_to(self.settings.data_dir)),
                    "mime_type": "application/json",
                    "preview_available": True,
                    "generated": "explicit",
                    "metadata": {
                        "step_id": "morphology",
                        "algorithm_key": alg("morphology", "image_2d.segment.morphology"),
                        "source_artifact_id": source_artifact_id,
                        "semantic_kind": "morphology_debug_json",
                        "coordinate_space": "image_pixel",
                        "effective_params": step_effective_params("morphology"),
                        **dict(state["morphology_debug_json"]),
                    },
                }
            )
        if isinstance(state.get("blob_contours_json"), list):
            payload_path = run_dir / "blob_contours.json"
            payload_path.write_text(json.dumps(state["blob_contours_json"], indent=2), encoding="utf-8")
            emitted.append(
                {
                    "artifact_id": "blob_contours",
                    "stage_id": "blob_detection",
                    "kind": "json",
                    "title": "Blob contours JSON",
                    "path": str(payload_path.relative_to(self.settings.data_dir)),
                    "mime_type": "application/json",
                    "preview_available": True,
                    "generated": "explicit",
                    "metadata": {
                        "step_id": "blob_detection",
                        "algorithm_key": alg("blob_detection", "image_2d.detect.blob_contours"),
                        "source_artifact_id": source_artifact_id,
                        "semantic_kind": "blob_contours",
                        "coordinate_space": "image_pixel",
                    },
                }
            )
        blob_metrics_payload = state.get("blob_metrics_summary_json")
        if not isinstance(blob_metrics_payload, dict):
            rows = state.get("blob_metrics_json")
            if isinstance(rows, list):
                blob_metrics_payload = {"candidates": rows}
        if isinstance(blob_metrics_payload, dict):
            blob_rows = blob_metrics_payload.get("candidates")
            if not isinstance(blob_rows, list):
                blob_rows = []
            rejected_count = len(state.get("blob_rejected_json") or []) if isinstance(state.get("blob_rejected_json"), list) else 0
            payload_path = run_dir / "blob_metrics.json"
            payload_path.write_text(json.dumps(blob_metrics_payload, indent=2), encoding="utf-8")
            emitted.append(
                {
                    "artifact_id": "blob_metrics",
                    "stage_id": "blob_detection",
                    "kind": "json",
                    "title": "Blob metrics JSON",
                    "path": str(payload_path.relative_to(self.settings.data_dir)),
                    "mime_type": "application/json",
                    "preview_available": True,
                    "generated": "explicit",
                    "metadata": {
                        "step_id": "blob_detection",
                        "algorithm_key": alg("blob_detection", "image_2d.detect.blob_contours"),
                        "source_artifact_id": source_artifact_id,
                        "semantic_kind": "blob_metrics",
                        "coordinate_space": "image_pixel",
                        "entries": blob_rows,
                        "rejected_count": int(blob_metrics_payload.get("rejected_count") or rejected_count),
                        "summary": blob_metrics_payload.get("stats"),
                        "params": blob_metrics_payload.get("params"),
                        "rejected_reason_counts": blob_metrics_payload.get("rejected_reason_counts"),
                    },
                }
            )
        if isinstance(state.get("blob_rejected_json"), list):
            payload_path = run_dir / "blob_rejected.json"
            payload_path.write_text(json.dumps(state["blob_rejected_json"], indent=2), encoding="utf-8")
            emitted.append(
                {
                    "artifact_id": "blob_rejected",
                    "stage_id": "blob_detection",
                    "kind": "json",
                    "title": "Rejected blob candidates JSON",
                    "path": str(payload_path.relative_to(self.settings.data_dir)),
                    "mime_type": "application/json",
                    "preview_available": True,
                    "generated": "explicit",
                    "metadata": {
                        "step_id": "blob_detection",
                        "algorithm_key": alg("blob_detection", "image_2d.detect.blob_contours"),
                        "source_artifact_id": source_artifact_id,
                        "semantic_kind": "blob_rejected",
                        "entries": state["blob_rejected_json"],
                    },
                }
            )
        if isinstance(state.get("ellipse_overlay_image"), np.ndarray):
            filename = "ellipse_overlay.png"
            out = run_dir / filename
            cv2.imwrite(str(out), state["ellipse_overlay_image"])
            emitted.append(
                {
                    "artifact_id": "ellipse_overlay",
                    "stage_id": "ellipse_fitting",
                    "kind": "image",
                    "title": "Ellipse overlay image",
                    "path": str(out.relative_to(self.settings.data_dir)),
                    "mime_type": "image/png",
                    "preview_available": True,
                    "generated": "explicit",
                    "metadata": {
                        "step_id": "ellipse_fitting",
                        "algorithm_key": alg("ellipse_fitting", "image_2d.measure.ellipse_fit"),
                        "source_artifact_id": source_artifact_id,
                        "semantic_kind": "ellipse_overlay",
                        "effective_params": step_effective_params("ellipse_fitting"),
                    },
                }
            )
        if isinstance(state.get("ellipse_debug_overlay_image"), np.ndarray):
            filename = "ellipse_debug_overlay.png"
            out = run_dir / filename
            cv2.imwrite(str(out), state["ellipse_debug_overlay_image"])
            emitted.append(
                {
                    "artifact_id": "ellipse_debug_overlay",
                    "stage_id": "ellipse_fitting",
                    "kind": "image",
                    "title": "Ellipse debug overlay image",
                    "path": str(out.relative_to(self.settings.data_dir)),
                    "mime_type": "image/png",
                    "preview_available": True,
                    "generated": "explicit",
                    "metadata": {
                        "step_id": "ellipse_fitting",
                        "algorithm_key": alg("ellipse_fitting", "image_2d.measure.ellipse_fit"),
                        "source_artifact_id": source_artifact_id,
                        "semantic_kind": "ellipse_debug_overlay",
                        "effective_params": step_effective_params("ellipse_fitting"),
                    },
                }
            )
        if isinstance(state.get("ellipse_metrics_json"), list):
            payload_path = run_dir / "ellipse_metrics.json"
            payload_path.write_text(json.dumps(state["ellipse_metrics_json"], indent=2), encoding="utf-8")
            emitted.append(
                {
                    "artifact_id": "ellipse_metrics",
                    "stage_id": "ellipse_fitting",
                    "kind": "json",
                    "title": "Ellipse metrics JSON",
                    "path": str(payload_path.relative_to(self.settings.data_dir)),
                    "mime_type": "application/json",
                    "preview_available": True,
                    "generated": "explicit",
                    "metadata": {
                        "step_id": "ellipse_fitting",
                        "algorithm_key": alg("ellipse_fitting", "image_2d.measure.ellipse_fit"),
                        "source_artifact_id": source_artifact_id,
                        "semantic_kind": "ellipse_metrics",
                        "effective_params": step_effective_params("ellipse_fitting"),
                        "entries": state["ellipse_metrics_json"],
                    },
                }
            )
        if isinstance(state.get("ellipse_candidates_json"), list):
            payload_path = run_dir / "ellipse_candidates.json"
            payload_path.write_text(json.dumps(state["ellipse_candidates_json"], indent=2), encoding="utf-8")
            emitted.append(
                {
                    "artifact_id": "ellipse_candidates",
                    "stage_id": "ellipse_fitting",
                    "kind": "json",
                    "title": "Ellipse candidates JSON",
                    "path": str(payload_path.relative_to(self.settings.data_dir)),
                    "mime_type": "application/json",
                    "preview_available": True,
                    "generated": "explicit",
                    "metadata": {
                        "step_id": "ellipse_fitting",
                        "algorithm_key": alg("ellipse_fitting", "image_2d.measure.ellipse_fit"),
                        "source_artifact_id": source_artifact_id,
                        "semantic_kind": "ellipse_candidates",
                        "effective_params": step_effective_params("ellipse_fitting"),
                        "entries": state["ellipse_candidates_json"],
                        "candidate_count": len(state["ellipse_candidates_json"]),
                    },
                }
            )
        if isinstance(state.get("ellipse_summary_json"), dict):
            payload_path = run_dir / "ellipse_summary.json"
            payload_path.write_text(json.dumps(state["ellipse_summary_json"], indent=2), encoding="utf-8")
            emitted.append(
                {
                    "artifact_id": "ellipse_summary",
                    "stage_id": "ellipse_fitting",
                    "kind": "json",
                    "title": "Ellipse summary JSON",
                    "path": str(payload_path.relative_to(self.settings.data_dir)),
                    "mime_type": "application/json",
                    "preview_available": True,
                    "generated": "explicit",
                    "metadata": {
                        "step_id": "ellipse_fitting",
                        "algorithm_key": alg("ellipse_fitting", "image_2d.measure.ellipse_fit"),
                        "source_artifact_id": source_artifact_id,
                        "semantic_kind": "ellipse_summary",
                        "effective_params": step_effective_params("ellipse_fitting"),
                        **dict(state["ellipse_summary_json"]),
                    },
                }
            )
        if isinstance(state.get("ellipse_debug_json"), dict):
            payload_path = run_dir / "ellipse_debug.json"
            payload_path.write_text(json.dumps(state["ellipse_debug_json"], indent=2), encoding="utf-8")
            emitted.append(
                {
                    "artifact_id": "ellipse_debug_json",
                    "stage_id": "ellipse_fitting",
                    "kind": "json",
                    "title": "Ellipse debug JSON",
                    "path": str(payload_path.relative_to(self.settings.data_dir)),
                    "mime_type": "application/json",
                    "preview_available": True,
                    "generated": "explicit",
                    "metadata": {
                        "step_id": "ellipse_fitting",
                        "algorithm_key": alg("ellipse_fitting", "image_2d.measure.ellipse_fit"),
                        "source_artifact_id": source_artifact_id,
                        "semantic_kind": "ellipse_debug_json",
                        "effective_params": step_effective_params("ellipse_fitting"),
                        **dict(state["ellipse_debug_json"]),
                    },
                }
            )
        if isinstance(state.get("classification_result_json"), dict):
            payload_path = run_dir / "classification_result.json"
            payload_path.write_text(json.dumps(state["classification_result_json"], indent=2), encoding="utf-8")
            emitted.append(
                {
                    "artifact_id": "classification_result",
                    "stage_id": "classification",
                    "kind": "json",
                    "title": "Classification result JSON",
                    "path": str(payload_path.relative_to(self.settings.data_dir)),
                    "mime_type": "application/json",
                    "preview_available": True,
                    "generated": "explicit",
                    "metadata": {
                        "step_id": "classification",
                        "algorithm_key": alg("classification", "image_2d.classify.ball_classifier"),
                        "source_artifact_id": source_artifact_id,
                        "semantic_kind": "classification_result",
                        "effective_params": step_effective_params("classification"),
                        **dict(state["classification_result_json"]),
                    },
                }
            )
        if isinstance(state.get("classification_overlay_objects"), list):
            overlay_meta = {
                "artifact_kind": "overlay",
                "overlay_type": "classification",
                "target_artifact_id": source_artifact_id,
                "overlay_coordinate_space": "image_pixel",
                "objects": state["classification_overlay_objects"],
            }
            payload_path = run_dir / "classification_overlay_metadata.json"
            payload_path.write_text(json.dumps(overlay_meta, indent=2), encoding="utf-8")
            emitted.append(
                {
                    "artifact_id": "classification_overlay_metadata",
                    "stage_id": "classification",
                    "kind": "json",
                    "title": "Classification overlay metadata",
                    "path": str(payload_path.relative_to(self.settings.data_dir)),
                    "mime_type": "application/json",
                    "preview_available": True,
                    "generated": "explicit",
                    "metadata": {
                        "step_id": "classification",
                        "algorithm_key": alg("classification", "image_2d.classify.ball_classifier"),
                        "source_artifact_id": source_artifact_id,
                        "semantic_kind": "classification_overlay_metadata",
                        **overlay_meta,
                    },
                }
            )
        if isinstance(state.get("classification_overlay_image"), np.ndarray):
            filename = "classification_overlay.png"
            out = run_dir / filename
            cv2.imwrite(str(out), state["classification_overlay_image"])
            emitted.append(
                {
                    "artifact_id": "classification_overlay_image",
                    "stage_id": "classification",
                    "kind": "image",
                    "title": "Classification overlay image",
                    "path": str(out.relative_to(self.settings.data_dir)),
                    "mime_type": "image/png",
                    "preview_available": True,
                    "generated": "explicit",
                    "metadata": {
                        "step_id": "classification",
                        "algorithm_key": alg("classification", "image_2d.classify.ball_classifier"),
                        "source_artifact_id": source_artifact_id,
                        "semantic_kind": "classification_overlay_image",
                        "overlay_type": "classification",
                        "target_artifact_id": source_artifact_id,
                        "overlay_coordinate_space": "image_pixel",
                    },
                }
            )
        return emitted

    def _render_overlay_preview(self, *, run_dir: Path, source_image: Any, overlays: list[dict[str, Any]]) -> str | None:
        if not isinstance(source_image, np.ndarray):
            return None
        canvas = source_image.copy()
        if canvas.ndim == 2:
            canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
        for overlay in overlays:
            if overlay.get("kind") != "overlay":
                continue
            geometry = overlay.get("geometry") if isinstance(overlay.get("geometry"), dict) else {}
            overlay_type = overlay.get("overlay_type")
            if overlay_type == "bbox":
                x, y = int(float(geometry.get("x", 0))), int(float(geometry.get("y", 0)))
                w, h = int(float(geometry.get("width", 0))), int(float(geometry.get("height", 0)))
                cv2.rectangle(canvas, (x, y), (x + w, y + h), (0, 255, 136), 2)
            elif overlay_type == "centroid":
                x, y = int(float(geometry.get("x", 0))), int(float(geometry.get("y", 0)))
                cv2.circle(canvas, (x, y), 4, (255, 138, 0), -1)
            elif overlay_type == "ellipse":
                cx, cy = int(float(geometry.get("cx", 0))), int(float(geometry.get("cy", 0)))
                rx, ry = int(float(geometry.get("rx", 0))), int(float(geometry.get("ry", 0)))
                angle = float(geometry.get("rotation", 0))
                cv2.ellipse(canvas, (cx, cy), (rx, ry), angle, 0, 360, (79, 140, 255), 2)
            elif overlay_type == "polyline":
                points = geometry.get("points")
                if isinstance(points, list) and points:
                    contour = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
                    cv2.polylines(canvas, [contour], True, (58, 134, 255), 2)
        filename = "overlay_preview_image.png"
        out = run_dir / filename
        cv2.imwrite(str(out), canvas)
        return str(out.relative_to(self.settings.data_dir))

    def promote_recipe(self, instance_id: str, *, run_id: str | None, recipe_type: str, notes: str | None = None) -> dict[str, Any]:
        instance = self._load_instance(instance_id)
        source = next((run for run in instance.execution_history if run.run_id == run_id), None) if run_id else None
        pipeline_snapshot = instance.model_dump(mode="json")
        pipeline_snapshot["pipeline_id"] = source.pipeline_id if source and source.pipeline_id else _pipeline_id_for_template(instance.template_id)
        recipe = RecipeVersion(
            id=f"recipe_{_new_id()}",
            pipeline_instance_id=instance.id,
            source_run_id=source.run_id if source else None,
            type=recipe_type,
            notes=notes,
            created_at=datetime.now(UTC),
            template_id=instance.template_id,
            pipeline_snapshot=pipeline_snapshot,
        )
        path = self.recipes_dir / f"{recipe.id}.json"
        path.write_text(json.dumps(recipe.model_dump(mode="json"), indent=2), encoding="utf-8")
        return recipe.model_dump(mode="json")

    def _load_instance(self, instance_id: str) -> PipelineInstance:
        path = self.instances_dir / f"{instance_id}.json"
        if not path.is_file():
            raise KeyError(f"Unknown pipeline instance: {instance_id}")
        return PipelineInstance.model_validate(json.loads(path.read_text(encoding="utf-8")))

    def _write_instance(self, instance: PipelineInstance) -> None:
        path = self.instances_dir / f"{instance.id}.json"
        path.write_text(json.dumps(instance.model_dump(mode="json"), indent=2), encoding="utf-8")

    def _load_recipe(self, recipe_version_id: str) -> RecipeVersion:
        path = self.recipes_dir / f"{recipe_version_id}.json"
        if not path.is_file():
            raise KeyError(f"Unknown recipe version: {recipe_version_id}")
        return RecipeVersion.model_validate(json.loads(path.read_text(encoding="utf-8")))


def _new_id() -> str:
    # A wall-clock timestamp is not unique under concurrent dispatch (two
    # threads can read the same microsecond), and process_run.run_id is a
    # global primary key that silently overwrites on collision -- see
    # storage/run_store.py. uuid4 has no realistic collision risk at any
    # scale this system reaches.
    return uuid.uuid4().hex


def _take_id_from_image_path(data_dir: Path, image_path: Path) -> str | None:
    try:
        rel = image_path.resolve().relative_to((data_dir / "incoming").resolve())
    except Exception:
        return None
    parts = rel.parts
    return parts[0] if parts else None


def _config_hash_for_instance(instance: PipelineInstance) -> str:
    payload = {
        "configured_steps": [step.model_dump(mode="json") for step in instance.configured_steps],
        "overrides": instance.overrides,
        "template_id": instance.template_id,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _pipeline_id_for_template(template_id: str) -> str | None:
    if template_id == "mining_steel_ball_classification_2d_reflectance_mvp":
        return "mining_steel_ball_classification_2d"
    return None
