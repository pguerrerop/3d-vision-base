from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from vision_3d_acquisition.contracts.modalities import CaptureModality, infer_modalities


class FileReferences(BaseModel):
    point_cloud: str | None = None
    point_cloud_npz: str | None = None
    height: str | None = None
    heightmap: str | None = None
    reflectance: str | None = None
    rgb: str | None = None
    rgb_video: str | None = None
    laser_rgb: str | None = None


class FrameSetAssets(BaseModel):
    point_cloud: str | None = None
    rgb: str | None = None
    laser_rgb: str | None = None
    reflectance: str | None = None
    rgb_video: str | None = None


class FrameSynchronization(BaseModel):
    mode: Literal["none", "timestamp", "hardware_trigger"] = "none"
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class FrameSet(BaseModel):
    frameset_id: str | None = None
    timestamp: str | None = None
    assets: FrameSetAssets = Field(default_factory=FrameSetAssets)
    synchronization: FrameSynchronization = Field(default_factory=FrameSynchronization)
    # Backward-compatible fields retained from the first contract revision.
    frame_count: int | None = Field(default=None, ge=1)
    synchronized: bool | None = None
    timestamp_source: str | None = None


class Units(BaseModel):
    x: str = "mm"
    y: str = "mm"
    z: str = "mm"


class Calibration(BaseModel):
    profile_distance_mm: float | None = None
    x_resolution_mm: float | None = None
    z_scale: float = 1.0
    z_offset: float = 0.0


class SensorInfo(BaseModel):
    model: str | None = None
    serial: str | None = None
    ip: str | None = None


class AcquisitionSessionMetadata(BaseModel):
    session_id: str
    operator: str | None = None
    sensor_setup: str | None = None
    acquisition_mode: str | None = None
    calibration_used: str | None = None
    conveyor_speed: float | None = None
    encoder_enabled: bool | None = None
    notes: str | None = None
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    take_ids: list[str] = Field(default_factory=list)


class AcquisitionMetadata(BaseModel):
    take_id: str
    source: str | dict[str, Any]
    mode: Literal["offline", "live"]
    created_at: str
    frame_count: int = Field(ge=1)
    modalities: list[CaptureModality] = Field(default_factory=list)
    session_id: str | None = None
    frameset: FrameSet | None = None
    files: FileReferences
    units: Units = Field(default_factory=Units)
    calibration: Calibration = Field(default_factory=Calibration)
    sensor: SensorInfo = Field(default_factory=SensorInfo)

    @field_validator("take_id")
    @classmethod
    def take_id_not_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("take_id must not be empty")
        return value

    @model_validator(mode="after")
    def populate_capture_model(self) -> "AcquisitionMetadata":
        payload = self.model_dump(mode="json")
        if not self.modalities:
            self.modalities = infer_modalities(payload)
        if self.frameset is None:
            self.frameset = FrameSet(
                frameset_id=f"{self.take_id}_fs0",
                timestamp=self.created_at,
                frame_count=self.frame_count,
                synchronized=False,
                timestamp_source="acquisition_metadata",
            )
        elif self.frameset.frame_count is None:
            self.frameset.frame_count = self.frame_count
        if self.frameset.frameset_id is None:
            self.frameset.frameset_id = f"{self.take_id}_fs0"
        if self.frameset.timestamp is None:
            self.frameset.timestamp = self.created_at
        if self.frameset.synchronized is None:
            self.frameset.synchronized = self.frameset.synchronization.mode != "none"
        if self.frameset.timestamp_source is None:
            self.frameset.timestamp_source = self.frameset.synchronization.mode
        if self.frameset.assets.point_cloud is None:
            self.frameset.assets.point_cloud = self.files.point_cloud
        if self.frameset.assets.rgb is None:
            self.frameset.assets.rgb = self.files.rgb
        if self.frameset.assets.rgb_video is None:
            self.frameset.assets.rgb_video = self.files.rgb_video
        if self.frameset.assets.laser_rgb is None:
            self.frameset.assets.laser_rgb = self.files.laser_rgb
        if self.frameset.assets.reflectance is None:
            self.frameset.assets.reflectance = self.files.reflectance
        return self
