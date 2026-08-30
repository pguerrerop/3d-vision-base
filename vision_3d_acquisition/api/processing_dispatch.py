from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any

from fastapi import HTTPException

from vision_3d_acquisition.api.filesystem import get_take_detail
from vision_3d_acquisition.api.histogram import resolve_source_image_path
from vision_3d_acquisition.api.pipeline_defaults_25d import merge_with_25d_defaults
from vision_3d_acquisition.api.settings import ApiSettings
from vision_3d_acquisition.apps.ball_inspection import run_ball_inspection_flow
from vision_3d_acquisition.apps.ball_inspection_25d import run_ball_inspection_25d_flow
from vision_3d_acquisition.pipelines.registry import list_pipelines
from vision_3d_acquisition.processing.pipeline import load_processing_config, process_take
from vision_3d_acquisition.processes.service import ProcessService
from vision_3d_acquisition.vision_core.serialization import write_result_json

PROCESS_SERVICE_PIPELINE_TO_TEMPLATE = {
    "mining_steel_ball_classification_2d": "mining_steel_ball_classification_2d_reflectance_mvp",
}
logger = logging.getLogger(__name__)


def dispatch_take_processing(
    *,
    settings: ApiSettings,
    take_id: str,
    pipeline_id: str,
    reprocess: bool,
    source_id: str | None = None,
    recipe_version_id: str | None = None,
    recipe_metadata: dict[str, Any] | None = None,
    acquisition_group_id: str | None = None,
    calibration_profile_id: str | None = None,
    stage_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    logger.info(
        "dispatch_take_processing take_id=%s pipeline_id=%s reprocess=%s source_id=%s recipe_version_id=%s acquisition_group_id=%s",
        take_id,
        pipeline_id,
        reprocess,
        source_id,
        recipe_version_id,
        acquisition_group_id,
    )
    take_dir = settings.incoming_dir / take_id
    if not take_dir.is_dir():
        raise HTTPException(status_code=404, detail="Take not found")
    detail = get_take_detail(settings, take_id)
    if detail is None:
        raise HTTPException(status_code=404, detail="Take not found")

    pipeline = next((item for item in list_pipelines() if item.get("id") == pipeline_id), None)
    if pipeline is None:
        raise HTTPException(status_code=404, detail=f"Unknown pipeline: {pipeline_id}")

    if pipeline.get("execution_backend") == "process_service":
        template_id = PROCESS_SERVICE_PIPELINE_TO_TEMPLATE.get(pipeline_id)
        if template_id is None:
            raise HTTPException(status_code=400, detail=f"No process template mapping configured for pipeline '{pipeline_id}'.")
        source_image, source_filename = resolve_source_image_path(settings, detail)
        if source_image is None:
            raise HTTPException(status_code=400, detail="No compatible source image found for this take.")
        service = ProcessService(settings)
        instances = service.list_instances()
        instance = next(
            (
                item
                for item in instances
                if item.get("template_id") == template_id and item.get("supported_input_type") == "rgb_image"
            ),
            None,
        )
        if instance is None:
            instance = service.create_instance(
                name="RGB Steel Ball Classification (Auto)",
                template_id=template_id,
                supported_input_type="rgb_image",
            )
        executed = service.execute(
            instance["id"],
            image_path=source_image,
            recipe_version_id=str(recipe_version_id) if recipe_version_id else None,
            pipeline_id=pipeline_id,
            source_id=source_id or None,
            take_id=take_id,
            acquisition_group_id=acquisition_group_id,
            calibration_profile_id=str(calibration_profile_id) if calibration_profile_id else None,
        )
        run = executed["run"]
        return {
            "ok": True,
            "take_id": take_id,
            "pipeline_id": pipeline_id,
            "pipeline_family": pipeline.get("pipeline_family"),
            "pipeline_instance_id": executed["pipeline_instance"]["id"],
            "run_id": run["run_id"],
            "status": run["status"],
            "source_image": source_filename,
            "result_path": str((settings.data_dir / "processes" / "runs" / executed["pipeline_instance"]["id"] / run["run_id"]).resolve()),
        }

    output_dir = settings.processed_dir / take_id
    if reprocess and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if pipeline_id == "3d_ball_inspection":
        result = run_ball_inspection_flow(
            settings.data_dir,
            take_id=take_id,
            recipe_version_id=str(recipe_version_id) if recipe_version_id else None,
            source_id=source_id or None,
            acquisition_group_id=acquisition_group_id,
            calibration_profile_id=str(calibration_profile_id) if calibration_profile_id else None,
        )
        return {
            "ok": True,
            "take_id": take_id,
            "pipeline_id": pipeline_id,
            "pipeline_family": pipeline.get("pipeline_family"),
            "status": result.result_payload.get("status"),
            "result_path": str(result.output_dir / "result.json"),
            "artifacts": [
                {"artifact_id": item.get("artifact_id"), "path": item.get("path")}
                for item in (result.result_payload.get("artifacts") or [])
                if isinstance(item, dict) and item.get("path")
            ],
            "result": result.result_payload,
        }

    if pipeline_id == "mining_steel_ball_classification_25d":
        merged_stage_params = merge_with_25d_defaults(settings.state_dir, stage_params)
        result = run_ball_inspection_25d_flow(
            settings.data_dir,
            take_id=take_id,
            recipe_version_id=str(recipe_version_id) if recipe_version_id else None,
            recipe_metadata=recipe_metadata,
            source_id=source_id or None,
            acquisition_group_id=acquisition_group_id,
            calibration_profile_id=str(calibration_profile_id) if calibration_profile_id else None,
            stage_params=merged_stage_params or None,
        )
        logger.info(
            "25d_processing_completed take_id=%s output_dir=%s status=%s artifacts=%s",
            take_id,
            str(result.output_dir),
            result.result_payload.get("status"),
            len(result.result_payload.get("artifacts") or []),
        )
        return {
            "ok": True,
            "take_id": take_id,
            "pipeline_id": pipeline_id,
            "pipeline_family": pipeline.get("pipeline_family"),
            "status": result.result_payload.get("status"),
            "result_path": str(result.output_dir / "result.json"),
            "artifacts": [
                {"artifact_id": item.get("artifact_id"), "path": item.get("path")}
                for item in (result.result_payload.get("artifacts") or [])
                if isinstance(item, dict) and item.get("path")
            ],
            "result": result.result_payload,
        }

    config = load_processing_config(Path("config/processing.yaml"))
    result = process_take(take_dir, output_dir, config)
    result.recipe_version_id = str(recipe_version_id) if recipe_version_id else None
    result.source_id = source_id or None
    result.acquisition_group_id = acquisition_group_id
    result.calibration_profile_id = str(calibration_profile_id) if calibration_profile_id else None
    write_result_json(output_dir / "result.json", result.model_dump(mode="json"))
    (output_dir / "DONE").touch()

    from vision_3d_acquisition.storage import catalog_sync  # noqa: WPS433 - avoids an import cycle

    catalog_sync.refresh_take(settings.data_dir, take_id)
    return {
        "ok": True,
        "take_id": take_id,
        "pipeline_id": pipeline_id,
        "pipeline_family": pipeline.get("pipeline_family"),
        "result_path": str(output_dir / "result.json"),
        "status": result.status,
    }
