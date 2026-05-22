from __future__ import annotations

from dataclasses import asdict
from typing import Any

from fastapi import HTTPException

from vision_3d_acquisition.acquisition.processing_contracts import (
    AcquiredTakeReady,
    AcquisitionProcessingDecision,
)
from vision_3d_acquisition.acquisition.processing_status import read_status, update_status
from vision_3d_acquisition.api.processing_dispatch import dispatch_take_processing
from vision_3d_acquisition.api.settings import ApiSettings
from vision_3d_acquisition.pipelines.registry import list_pipelines
from vision_3d_acquisition.processes.bindings import ProcessBindingService


def normalize_processing_modality(modality: str) -> tuple[str, str | None]:
    raw = str(modality or "").strip()
    if raw == "heightmap_2_5d":
        return "heightmap", raw
    return raw, None


def process_acquired_take_if_bound(
    settings: ApiSettings,
    acquired_take: AcquiredTakeReady,
    *,
    purpose: str = "acquisition_inspection",
    auto_process: bool = True,
) -> AcquisitionProcessingDecision:
    normalized_modality, original_alias = normalize_processing_modality(acquired_take.modality)
    metadata = dict(acquired_take.metadata or {})
    if original_alias:
        metadata.setdefault("original_modality", original_alias)

    base_status = {
        "take_id": acquired_take.take_id,
        "source_id": acquired_take.source_id,
        "modality": normalized_modality,
        "modality_family": acquired_take.modality_family,
        "acquisition_group_id": acquired_take.acquisition_group_id,
        "acquisition_process_id": acquired_take.acquisition_process_id,
        "acquisition_run_id": acquired_take.acquisition_run_id,
        "warnings": list(acquired_take.warnings or []),
        "metadata": metadata,
    }
    update_status(settings.data_dir, acquired_take.take_id, {**base_status, "status": "ready_for_processing"})

    if not auto_process:
        update_status(settings.data_dir, acquired_take.take_id, {"status": "ready_not_processed"})
        return AcquisitionProcessingDecision(
            ok=True,
            status="ready_not_processed",
            normalized_modality=normalized_modality,
            original_modality=acquired_take.modality,
            warnings=list(acquired_take.warnings or []),
            metadata=metadata,
        )

    try:
        binding = ProcessBindingService(settings).resolve_process_binding(acquired_take.source_id, normalized_modality, purpose)  # type: ignore[arg-type]
    except ValueError as exc:
        update_status(settings.data_dir, acquired_take.take_id, {"status": "processing_failed", "warnings": [str(exc)]})
        return AcquisitionProcessingDecision(
            ok=False,
            status="processing_failed",
            normalized_modality=normalized_modality,
            original_modality=acquired_take.modality,
            warning=str(exc),
            warnings=[str(exc)],
            metadata=metadata,
        )

    if binding is None:
        warning = "No active processing binding found for this source/modality/purpose."
        update_status(settings.data_dir, acquired_take.take_id, {"status": "processing_binding_missing", "warnings": [warning]})
        return AcquisitionProcessingDecision(
            ok=True,
            status="processing_binding_missing",
            normalized_modality=normalized_modality,
            original_modality=acquired_take.modality,
            warning=warning,
            warnings=[warning],
            metadata=metadata,
        )

    pipeline_id = str(binding.get("pipeline_id") or "")
    recipe_version_id = str(binding.get("active_recipe_version_id") or "") or None
    calibration_profile_id = str(binding.get("calibration_profile_id") or "") or None

    pipeline = next((item for item in list_pipelines() if str(item.get("id") or "") == pipeline_id), None)
    if pipeline is None:
        warning = f"Unknown pipeline from binding: {pipeline_id}"
        update_status(settings.data_dir, acquired_take.take_id, {"status": "processing_failed", "warnings": [warning]})
        return AcquisitionProcessingDecision(
            ok=False,
            status="processing_failed",
            normalized_modality=normalized_modality,
            original_modality=acquired_take.modality,
            warning=warning,
            warnings=[warning],
            pipeline_id=pipeline_id,
            recipe_version_id=recipe_version_id,
            calibration_profile_id=calibration_profile_id,
            metadata=metadata,
        )

    update_status(settings.data_dir, acquired_take.take_id, {"status": "processing_enqueued", "pipeline_id": pipeline_id, "recipe_version_id": recipe_version_id})
    update_status(settings.data_dir, acquired_take.take_id, {"status": "processing_running"})

    try:
        run_ref = dispatch_take_processing(
            settings=settings,
            take_id=acquired_take.take_id,
            pipeline_id=pipeline_id,
            reprocess=True,
            source_id=acquired_take.source_id,
            recipe_version_id=recipe_version_id,
            acquisition_group_id=acquired_take.acquisition_group_id,
            calibration_profile_id=calibration_profile_id,
        )
    except HTTPException as exc:
        message = str(exc.detail)
        update_status(settings.data_dir, acquired_take.take_id, {"status": "processing_failed", "warnings": [message]})
        return AcquisitionProcessingDecision(
            ok=False,
            status="processing_failed",
            normalized_modality=normalized_modality,
            original_modality=acquired_take.modality,
            warning=message,
            warnings=[message],
            pipeline_id=pipeline_id,
            recipe_version_id=recipe_version_id,
            calibration_profile_id=calibration_profile_id,
            metadata=metadata,
        )
    except Exception as exc:
        message = f"Bound processing failed: {exc}"
        update_status(settings.data_dir, acquired_take.take_id, {"status": "processing_failed", "warnings": [message]})
        return AcquisitionProcessingDecision(
            ok=False,
            status="processing_failed",
            normalized_modality=normalized_modality,
            original_modality=acquired_take.modality,
            warning=message,
            warnings=[message],
            pipeline_id=pipeline_id,
            recipe_version_id=recipe_version_id,
            calibration_profile_id=calibration_profile_id,
            metadata=metadata,
        )

    run_metadata = {
        "take_id": acquired_take.take_id,
        "source_id": acquired_take.source_id,
        "acquisition_group_id": acquired_take.acquisition_group_id,
        "acquisition_process_id": acquired_take.acquisition_process_id,
        "acquisition_run_id": acquired_take.acquisition_run_id,
        "pipeline_id": pipeline_id,
        "recipe_version_id": recipe_version_id,
        "config_snapshot_hash": (run_ref.get("result") or {}).get("config_snapshot_hash") or run_ref.get("config_snapshot_hash"),
        "modality": normalized_modality,
        "modality_family": acquired_take.modality_family,
    }
    update_status(
        settings.data_dir,
        acquired_take.take_id,
        {
            "status": "processing_completed" if run_ref.get("ok") else "processing_failed",
            "pipeline_id": pipeline_id,
            "recipe_version_id": recipe_version_id,
            "calibration_profile_id": calibration_profile_id,
            "run_ref": run_ref,
            "run_metadata": run_metadata,
        },
    )

    return AcquisitionProcessingDecision(
        ok=bool(run_ref.get("ok")),
        status="processing_completed" if run_ref.get("ok") else "processing_failed",
        normalized_modality=normalized_modality,
        original_modality=acquired_take.modality,
        warnings=list(acquired_take.warnings or []),
        pipeline_id=pipeline_id,
        recipe_version_id=recipe_version_id,
        calibration_profile_id=calibration_profile_id,
        run_ref=run_ref,
        metadata=metadata,
    )


def trigger_bound_processing_for_take(
    settings: ApiSettings,
    *,
    take_id: str,
    source_id: str,
    modality: str,
    purpose: str = "acquisition_inspection",
    modality_family: str | None = None,
    acquisition_group_id: str | None = None,
    acquisition_process_id: str | None = None,
    acquisition_run_id: str | None = None,
    metadata: dict[str, Any] | None = None,
    auto_process: bool = True,
) -> dict[str, Any]:
    decision = process_acquired_take_if_bound(
        settings,
        AcquiredTakeReady(
            take_id=take_id,
            source_id=source_id,
            modality=modality,
            modality_family=modality_family,
            acquisition_group_id=acquisition_group_id,
            acquisition_process_id=acquisition_process_id,
            acquisition_run_id=acquisition_run_id,
            metadata=metadata or {},
        ),
        purpose=purpose,
        auto_process=auto_process,
    )
    payload = asdict(decision)
    if decision.warning:
        payload["warning"] = decision.warning
    return payload


def acquisition_processing_status(settings: ApiSettings, take_id: str) -> dict[str, Any]:
    return read_status(settings.data_dir, take_id) or {"take_id": take_id, "status": "acquired"}
