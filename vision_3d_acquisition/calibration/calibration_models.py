from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from vision_3d_acquisition.contracts.modalities import CaptureModality


PlaneLabel = Literal["belt", "outer_plane_ignore", "unused"]
Camera2DTargetType = Literal["charuco", "checkerboard"]


class ObjectFilterConfig(BaseModel):
    min_height_above_belt_mm: float = 3.0
    max_height_above_belt_mm: float = 130.0
    require_center_inside_belt: bool = True
    min_fraction_points_inside_belt: float = Field(default=0.6, ge=0.0, le=1.0)


class PlaneCalibration(BaseModel):
    plane_id: str
    label: PlaneLabel
    plane_model: tuple[float, float, float, float]
    point_count: int = Field(ge=0)
    bbox_min_mm: tuple[float, float, float]
    bbox_max_mm: tuple[float, float, float]
    extent_mm: tuple[float, float, float]
    avg_z_mm: float
    normal: tuple[float, float, float] | None = None
    roi_polygon_xy_mm: list[tuple[float, float]] = Field(default_factory=list)


class HeightmapCalibration(BaseModel):
    x_resolution_mm: float = Field(gt=0)
    y_resolution_mm: float = Field(gt=0)
    z_scale: float = 1.0
    z_offset: float = 0.0
    coordinate_system: str = "sensor_xy_z_mm"
    tilt_correction: dict[str, float] | None = None
    belt_reference_plane: tuple[float, float, float, float] | None = None
    roi: dict[str, object] | None = None


class Camera2DTargetConfig(BaseModel):
    type: Camera2DTargetType = "charuco"
    squares_x: int = Field(default=7, ge=2)
    squares_y: int = Field(default=5, ge=2)
    square_length_mm: float = Field(default=25.0, gt=0)
    marker_length_mm: float = Field(default=18.0, gt=0)
    dictionary: str = "DICT_4X4_50"

    @field_validator("dictionary")
    @classmethod
    def validate_dictionary(cls, value: str) -> str:
        allowed = {"DICT_4X4_50", "DICT_4X4_100", "DICT_5X5_50", "DICT_5X5_100", "DICT_6X6_250"}
        if value not in allowed:
            raise ValueError(f"Unsupported dictionary: {value}")
        return value

    @model_validator(mode="after")
    def validate_board_dimensions(self) -> "Camera2DTargetConfig":
        if self.marker_length_mm >= self.square_length_mm:
            raise ValueError("marker_length_mm must be smaller than square_length_mm")
        return self


class Camera2DIntrinsics(BaseModel):
    camera_matrix: list[list[float]]
    dist_coeffs: list[float]
    reprojection_error: float = Field(ge=0)
    image_width: int = Field(ge=1)
    image_height: int = Field(ge=1)


class Camera2DBeltPlane(BaseModel):
    homography: list[list[float]]
    mm_per_px_x: float = Field(gt=0)
    mm_per_px_y: float = Field(gt=0)
    reference_plane_z_mm: float = 0.0


class SystemCalibration(BaseModel):
    calibration_id: str
    calibration_type: str = "plane_3d"
    source_modalities: list[CaptureModality] = Field(default_factory=lambda: ["point_cloud"])
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    # plane_3d payload
    reference_take_id: str | None = None
    planes: list[PlaneCalibration] = Field(default_factory=list)
    object_filter: ObjectFilterConfig = Field(default_factory=ObjectFilterConfig)
    heightmap: HeightmapCalibration | None = None

    # camera_2d payload
    source_id: str | None = None
    target: Camera2DTargetConfig | None = None
    intrinsics: Camera2DIntrinsics | None = None
    belt_plane: Camera2DBeltPlane | None = None
    camera_runtime_settings: dict[str, float | bool | None] | None = None

    @field_validator("calibration_id")
    @classmethod
    def non_empty_calibration_id(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be empty")
        return value

    @field_validator("reference_take_id")
    @classmethod
    def non_empty_reference_take_id(cls, value: str | None) -> str | None:
        if value is None:
            return value
        if not value.strip():
            raise ValueError("must not be empty")
        return value

    @model_validator(mode="after")
    def validate_by_type(self) -> "SystemCalibration":
        if self.calibration_type == "plane_3d":
            if "point_cloud" not in self.source_modalities:
                raise ValueError("plane_3d calibration requires point_cloud source modality")
            if not self.reference_take_id:
                raise ValueError("plane_3d calibration requires reference_take_id")
            belt_count = sum(1 for plane in self.planes if plane.label == "belt")
            if belt_count != 1:
                raise ValueError("calibration must contain exactly one belt plane")
        elif self.calibration_type == "camera_2d":
            if "rgb" not in self.source_modalities:
                raise ValueError("camera_2d calibration requires rgb source modality")
            if not self.source_id:
                raise ValueError("camera_2d calibration requires source_id")
            if self.target is None:
                raise ValueError("camera_2d calibration requires target")
            if self.intrinsics is None:
                raise ValueError("camera_2d calibration requires intrinsics")
            if self.belt_plane is None:
                raise ValueError("camera_2d calibration requires belt_plane")
        return self

    def belt_plane_3d(self) -> PlaneCalibration:
        return next(plane for plane in self.planes if plane.label == "belt")

    def belt_plane(self) -> PlaneCalibration:
        return self.belt_plane_3d()

    def ignored_planes(self) -> list[PlaneCalibration]:
        return [plane for plane in self.planes if plane.label == "outer_plane_ignore"]
