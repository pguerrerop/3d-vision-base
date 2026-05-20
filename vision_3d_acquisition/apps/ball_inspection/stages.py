from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from vision_3d_acquisition.vision_core.pipelines.core import PipelineContext


@dataclass
class FitSphereOrEllipseStage:
    """Placeholder stage to normalize shape-fit features for classification."""

    name: str = "FitSphereOrEllipseStage"
    category: str = "measurement"

    def run(self, context: PipelineContext) -> None:
        objects = context.get_artifact("objects", [])
        for item in objects:
            if item.get("diameter_mm") is None and item.get("diameter_estimate_mm") is not None:
                item["diameter_mm"] = item["diameter_estimate_mm"]
        context.set_artifact("objects", objects)


@dataclass
class BallClassificationStage:
    """Domain-specific steel-ball classification heuristics for the POC."""

    name: str = "BallClassificationStage"
    category: str = "classification"

    def run(self, context: PipelineContext) -> None:
        objects = context.get_artifact("objects", [])
        classified: list[str] = []
        for item in objects:
            class_name, confidence = _classify_object(item)
            item["class_name"] = class_name
            item["confidence"] = confidence
            classified.append(class_name)
            object_id = item.get("object_id")
            if isinstance(object_id, int):
                context.add_processing_artifact(
                    artifact_id=f"classification_object_{object_id}",
                    stage_id="classification",
                    object_id=object_id,
                    kind="json",
                    title=f"Object #{object_id} classification",
                    metadata={"source": "native_pipeline", "producer": self.name, "modality": "point_cloud"},
                )
        context.metrics["classified_objects"] = len(classified)
        context.metrics["ball_candidates"] = classified.count("ball")
        context.set_artifact("objects", objects)


@dataclass
class StatisticsStage:
    """Build aggregate decision statistics from classified objects."""

    name: str = "StatisticsStage"
    category: str = "classification"

    def run(self, context: PipelineContext) -> None:
        objects = context.get_artifact("objects", [])
        ball_count = sum(1 for item in objects if item.get("class_name") == "ball")
        non_ball_count = sum(1 for item in objects if item.get("class_name") == "non_ball")
        unknown_count = sum(1 for item in objects if item.get("class_name") == "unknown")
        object_count = len(objects)
        decision = "accept"
        if non_ball_count > 0:
            decision = "review"
        if object_count == 0:
            decision = "review"
        confidence = None
        if object_count > 0:
            confidence = round(ball_count / max(1, object_count), 3)
        context.set_artifact(
            "summary",
            {
                "object_count": object_count,
                "ball_count": ball_count,
                "non_ball_count": non_ball_count,
                "decision": decision,
                "confidence": confidence,
            },
        )
        context.set_artifact("algorithm_stage", "classification")
        context.metrics["unknown_objects"] = unknown_count


def _classify_object(item: dict[str, Any]) -> tuple[str, float | None]:
    point_count = int(item.get("point_count") or 0)
    diameter = item.get("diameter_mm") or item.get("diameter_estimate_mm")
    dimensions = item.get("dimensions_mm")
    if diameter is not None:
        diameter_value = float(diameter)
        if 5.0 <= diameter_value <= 120.0:
            return "ball", 0.72
        return "non_ball", 0.68
    if isinstance(dimensions, (list, tuple)) and len(dimensions) == 3:
        dims = [float(value) for value in dimensions]
        min_dim = max(0.001, min(dims))
        ratio = max(dims) / min_dim
        if ratio <= 1.35 and point_count >= 20:
            return "ball", 0.66
        return "non_ball", 0.61
    return "unknown", None
