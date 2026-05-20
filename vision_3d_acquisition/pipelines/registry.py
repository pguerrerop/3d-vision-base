from __future__ import annotations

from typing import Any


def _stage(
    *,
    stage_id: str,
    display_name: str,
    description: str,
    required_modalities: list[str],
    optional_modalities: list[str],
    produced_artifact_kinds: list[str],
    object_outputs: bool,
    supports_real_time: bool,
    dependencies: list[str] | None = None,
    optional_stage: bool = False,
    condition: str | None = None,
    version: str = "1.0",
    implemented: bool = True,
) -> dict[str, Any]:
    return {
        "id": stage_id,
        "stage_id": stage_id,
        "display_name": display_name,
        "version": version,
        "description": description,
        "required_modalities": required_modalities,
        "optional_modalities": optional_modalities,
        "produced_artifact_kinds": produced_artifact_kinds,
        "object_outputs": object_outputs,
        "supports_real_time": supports_real_time,
        "dependencies": dependencies or [],
        "optional_stage": optional_stage,
        "condition": condition,
        "implemented": implemented,
    }


PIPELINES: list[dict[str, Any]] = [
    {
        "id": "mining_steel_ball_classification_2d",
        "name": "mining_steel_ball_classification_2d",
        "display_name": "Mining Steel Ball Classification (RGB/2D MVP)",
        "pipeline_family": "2d",
        "execution_backend": "process_service",
        "supported_modalities": ["rgb", "reflectance"],
        "supports_live_processing": False,
        "supports_batch": True,
        "supports_partial_stages": False,
        "required_modalities": ["rgb"],
        "optional_modalities": ["reflectance"],
        "implemented": True,
        "description": "Template-driven 2D reflectance workflow with thresholding, morphology cleanup, blob detection, ellipse fitting, and classification.",
        "stages": [
            _stage(stage_id="input", display_name="Input image", description="Loads RGB/grayscale/reflectance image.", required_modalities=["rgb"], optional_modalities=["reflectance"], produced_artifact_kinds=["image"], object_outputs=False, supports_real_time=True),
            _stage(stage_id="segmentation", display_name="Threshold + morphology", description="Binary threshold and morphology cleanup.", required_modalities=["reflectance"], optional_modalities=["rgb"], produced_artifact_kinds=["image", "overlay"], object_outputs=False, supports_real_time=True, dependencies=["input"]),
            _stage(stage_id="detection", display_name="Blob/contour detection", description="Connected contours, regions, bounding boxes, and centroids.", required_modalities=["reflectance"], optional_modalities=["rgb"], produced_artifact_kinds=["overlay", "table"], object_outputs=True, supports_real_time=True, dependencies=["segmentation"]),
            _stage(stage_id="measurement", display_name="Ellipse fitting + metrics", description="Ellipse fitting and circularity/eccentricity/diameter extraction.", required_modalities=["reflectance"], optional_modalities=["rgb"], produced_artifact_kinds=["overlay", "table", "metric"], object_outputs=True, supports_real_time=True, dependencies=["detection"]),
            _stage(stage_id="classification", display_name="Classification + summary", description="Ball/non-ball and spherical-enough classification with summary outputs.", required_modalities=["reflectance"], optional_modalities=["rgb"], produced_artifact_kinds=["overlay", "table", "json"], object_outputs=True, supports_real_time=True, dependencies=["measurement"]),
        ],
        "composition": {
            "execution_order": ["input", "segmentation", "detection", "measurement", "classification"],
            "artifact_flow": {
                "input": ["source image artifact"],
                "segmentation": ["threshold mask", "morphology mask"],
                "detection": ["contours", "bounding boxes", "centroids"],
                "measurement": ["ellipse fits", "measurement table"],
                "classification": ["classification table", "result summary"],
            },
            "conditional_stages": [],
            "optional_stages": [],
        },
    },
    {
        "id": "3d_ball_inspection",
        "name": "3d_ball_inspection",
        "display_name": "3D Ball Inspection",
        "pipeline_family": "3d",
        "execution_backend": "filesystem_queue",
        "supported_modalities": ["point_cloud", "rgb"],
        "supports_live_processing": True,
        "supports_batch": True,
        "supports_partial_stages": True,
        "required_modalities": ["point_cloud"],
        "optional_modalities": ["rgb"],
        "implemented": True,
        "description": "Point-cloud segmentation, measurement, and ball-oriented classification/statistics.",
        "stages": [
            _stage(
                stage_id="segmentation",
                display_name="Object segmentation",
                description="Segments foreground clusters and candidate object regions from the point cloud.",
                required_modalities=["point_cloud"],
                optional_modalities=["rgb"],
                produced_artifact_kinds=["image", "overlay", "table"],
                object_outputs=True,
                supports_real_time=True,
            ),
            _stage(
                stage_id="classification",
                display_name="Ball classification",
                description="Applies ball/non-ball heuristics and emits confidence overlays.",
                required_modalities=["point_cloud"],
                optional_modalities=["rgb"],
                produced_artifact_kinds=["json", "table", "overlay", "text"],
                object_outputs=True,
                supports_real_time=True,
                dependencies=["segmentation"],
            ),
            _stage(
                stage_id="measurement",
                display_name="Diameter/statistics",
                description="Produces object measurements, provenance annotations, and summary metrics.",
                required_modalities=["point_cloud"],
                optional_modalities=["rgb"],
                produced_artifact_kinds=["metric", "table", "overlay", "text"],
                object_outputs=True,
                supports_real_time=True,
                dependencies=["classification"],
            ),
        ],
        "composition": {
            "execution_order": ["segmentation", "classification", "measurement"],
            "artifact_flow": {
                "segmentation": ["foreground_clusters", "segmentation overlays"],
                "classification": ["classification_table", "classification overlays"],
                "measurement": ["measurement_table", "measurement overlays", "measurement_summary"],
            },
            "conditional_stages": [],
            "optional_stages": [],
        },
    },
    {
        "id": "rgb_segmentation",
        "name": "rgb_segmentation",
        "display_name": "RGB Segmentation",
        "pipeline_family": "2d",
        "execution_backend": "process_service",
        "supported_modalities": ["rgb"],
        "supports_live_processing": False,
        "supports_batch": True,
        "supports_partial_stages": False,
        "required_modalities": ["rgb"],
        "optional_modalities": [],
        "implemented": False,
        "description": "Future 2D RGB segmentation pipeline.",
        "stages": [
            _stage(
                stage_id="segmentation",
                display_name="RGB mask segmentation",
                description="Future RGB segmentation masks and contour generation.",
                required_modalities=["rgb"],
                optional_modalities=[],
                produced_artifact_kinds=["image", "overlay"],
                object_outputs=True,
                supports_real_time=True,
                implemented=False,
            ),
            _stage(
                stage_id="artifact_export",
                display_name="Mask/artifact export",
                description="Future export stage for RGB segmentation artifacts.",
                required_modalities=["rgb"],
                optional_modalities=[],
                produced_artifact_kinds=["file", "json"],
                object_outputs=False,
                supports_real_time=False,
                dependencies=["segmentation"],
                implemented=False,
            ),
        ],
        "composition": {
            "execution_order": ["segmentation", "artifact_export"],
            "artifact_flow": {"segmentation": ["rgb masks"], "artifact_export": ["export bundle"]},
            "conditional_stages": [],
            "optional_stages": [],
        },
    },
    {
        "id": "rgb_ball_classifier",
        "name": "rgb_ball_classifier",
        "display_name": "RGB Ball Classification",
        "pipeline_family": "2d",
        "execution_backend": "process_service",
        "supported_modalities": ["rgb"],
        "supports_live_processing": False,
        "supports_batch": True,
        "supports_partial_stages": False,
        "required_modalities": ["rgb"],
        "optional_modalities": [],
        "implemented": False,
        "description": "Future RGB-only ball classifier.",
        "stages": [
            _stage(
                stage_id="classification",
                display_name="2D ball classification",
                description="Future RGB-only classifier with 2D overlays.",
                required_modalities=["rgb"],
                optional_modalities=[],
                produced_artifact_kinds=["json", "overlay", "table"],
                object_outputs=True,
                supports_real_time=True,
                implemented=False,
            ),
            _stage(
                stage_id="statistics",
                display_name="Classification statistics",
                description="Future aggregate stage for 2D classification metrics.",
                required_modalities=["rgb"],
                optional_modalities=[],
                produced_artifact_kinds=["metric", "table"],
                object_outputs=False,
                supports_real_time=True,
                dependencies=["classification"],
                implemented=False,
            ),
        ],
        "composition": {
            "execution_order": ["classification", "statistics"],
            "artifact_flow": {"classification": ["2d overlays"], "statistics": ["summary metrics"]},
            "conditional_stages": [],
            "optional_stages": [],
        },
    },
    {
        "id": "2d_3d_fusion",
        "name": "2d_3d_fusion",
        "display_name": "2D/3D Fusion",
        "pipeline_family": "generic",
        "execution_backend": "future",
        "supported_modalities": ["point_cloud", "rgb", "rgb_video"],
        "supports_live_processing": False,
        "supports_batch": True,
        "supports_partial_stages": False,
        "required_modalities": ["point_cloud", "rgb"],
        "optional_modalities": ["rgb_video"],
        "implemented": False,
        "description": "Future synchronized RGB plus 3D fusion workflow.",
        "stages": [
            _stage(
                stage_id="registration",
                display_name="RGB/3D registration",
                description="Future multimodal registration stage.",
                required_modalities=["point_cloud", "rgb"],
                optional_modalities=["rgb_video"],
                produced_artifact_kinds=["json", "overlay"],
                object_outputs=False,
                supports_real_time=False,
                implemented=False,
            ),
            _stage(
                stage_id="fusion",
                display_name="Fused evidence",
                description="Future evidence fusion stage for synchronized modalities.",
                required_modalities=["point_cloud", "rgb"],
                optional_modalities=["rgb_video"],
                produced_artifact_kinds=["image", "overlay", "table", "point_cloud"],
                object_outputs=True,
                supports_real_time=False,
                dependencies=["registration"],
                implemented=False,
            ),
            _stage(
                stage_id="classification",
                display_name="Fused classification",
                description="Future classification from fused representations.",
                required_modalities=["point_cloud", "rgb"],
                optional_modalities=["rgb_video"],
                produced_artifact_kinds=["json", "metric", "overlay"],
                object_outputs=True,
                supports_real_time=False,
                dependencies=["fusion"],
                implemented=False,
            ),
        ],
        "composition": {
            "execution_order": ["registration", "fusion", "classification"],
            "artifact_flow": {
                "registration": ["projection references"],
                "fusion": ["fused artifacts"],
                "classification": ["fusion classifications"],
            },
            "conditional_stages": ["registration"],
            "optional_stages": ["rgb_video alignment refinement"],
        },
    },
]


def list_pipelines() -> list[dict[str, Any]]:
    return [
        dict(
            pipeline,
            stages=[dict(stage) for stage in pipeline["stages"]],
            composition=dict(pipeline.get("composition") or {}),
        )
        for pipeline in PIPELINES
    ]


def get_pipeline(pipeline_id: str) -> dict[str, Any] | None:
    for pipeline in list_pipelines():
        if pipeline["id"] == pipeline_id or pipeline["name"] == pipeline_id:
            return pipeline
    return None


def default_pipeline_info(pipeline_id: str = "3d_ball_inspection") -> dict[str, Any]:
    pipeline = get_pipeline(pipeline_id)
    if pipeline is None:
        raise KeyError(f"Unknown pipeline: {pipeline_id}")
    return pipeline


def list_processing_units(pipeline_id: str = "3d_ball_inspection") -> list[dict[str, Any]]:
    pipeline = default_pipeline_info(pipeline_id)
    return [dict(stage) for stage in pipeline["stages"]]


def get_processing_unit(stage_id: str, pipeline_id: str = "3d_ball_inspection") -> dict[str, Any] | None:
    for stage in list_processing_units(pipeline_id):
        if stage.get("id") == stage_id or stage.get("stage_id") == stage_id:
            return stage
    return None


def build_stage_outputs(files: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "stage": "source_decode",
            "display_name": "Raw input decode",
            "artifacts": _non_empty(
                {
                    "point_cloud": files.get("point_cloud"),
                    "point_cloud_npz": files.get("point_cloud_npz"),
                    "input_preview": files.get("input_preview"),
                }
            ),
        },
        {
            "stage": "segmentation",
            "display_name": "Object segmentation",
            "artifacts": _non_empty(
                {
                    "plane_segmentation": files.get("debug_plane_segmentation"),
                    "foreground": files.get("debug_foreground"),
                    "clusters": files.get("debug_clusters"),
                    "calibrated_planes": files.get("debug_calibrated_planes"),
                    "filtered_foreground": files.get("debug_filtered_foreground"),
                    "rejected_points": files.get("debug_rejected_points"),
                    "filtered_clusters": files.get("debug_clusters_filtered"),
                }
            ),
        },
        {
            "stage": "classification",
            "display_name": "Ball classification",
            "artifacts": _non_empty({"overlay": files.get("overlay")}),
        },
        {
            "stage": "measurement",
            "display_name": "Diameter/statistics",
            "artifacts": {},
        },
        {
            "stage": "fusion",
            "display_name": "Fusion",
            "implemented": False,
            "artifacts": {},
        },
    ]


def build_stage_outputs_from_artifacts(artifacts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_stage: dict[str, list[dict[str, Any]]] = {}
    for artifact in artifacts:
        stage_id = str(artifact.get("stage_id") or "result")
        by_stage.setdefault(stage_id, []).append(artifact)
    outputs: list[dict[str, Any]] = []
    for stage_id, items in by_stage.items():
        outputs.append(
            {
                "stage": stage_id,
                "display_name": stage_id.replace("_", " ").title(),
                "artifacts": {item["artifact_id"]: item.get("path") for item in items},
                "artifact_ids": [item["artifact_id"] for item in items],
            }
        )
    if not any(item["stage"] == "fusion" for item in outputs):
        outputs.append({"stage": "fusion", "display_name": "Fusion", "implemented": False, "artifacts": {}, "artifact_ids": []})
    return outputs


def _non_empty(values: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in values.items() if value}
