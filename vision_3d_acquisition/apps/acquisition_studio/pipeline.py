from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from vision_3d_acquisition.vision_core.acquisition.sources import FileSource
from vision_3d_acquisition.vision_core.pipelines import PipelineContext, PipelineRunner
from vision_3d_acquisition.vision_core.pipelines.stages import (
    ApplyCalibrationStage,
    DecodeHeightmapStage,
    LoadCaptureStage,
    PlaneFilterStage,
    SegmentObjectsStage,
)


@dataclass
class AcquisitionDebugResult:
    take_id: str
    output_dir: Path
    profiling: dict
    cluster_count: int
    plane_model: tuple[float, float, float, float] | None
    input_stats: dict


def run_acquisition_debug_flow(
    data_dir: Path,
    *,
    take_id: str | None = None,
    calibration: Path | None = None,
    prefer_fast_cloud: bool = True,
    render_preview: bool = True,
) -> AcquisitionDebugResult:
    source = FileSource(data_dir)
    output_root = data_dir / "processed"
    context = PipelineContext()
    runner = PipelineRunner(
        stages=[
            LoadCaptureStage(source=source, take_id=take_id, output_root=output_root),
            DecodeHeightmapStage(prefer_fast_cloud=prefer_fast_cloud, render_preview=render_preview),
            ApplyCalibrationStage(calibration_path=calibration),
            PlaneFilterStage(),
            SegmentObjectsStage(),
        ]
    )
    result = runner.run(context)
    return AcquisitionDebugResult(
        take_id=str(result.artifacts["take_id"]),
        output_dir=result.artifacts["output_dir"],
        profiling=result.profiling,
        cluster_count=int(result.metrics.get("cluster_count", 0)),
        plane_model=result.artifacts.get("plane_model"),
        input_stats=result.artifacts.get("input_stats", {}),
    )
