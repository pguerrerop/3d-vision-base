from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from vision_3d_acquisition.contracts.modalities import CaptureModality, assets_by_modality, infer_modalities


@dataclass(frozen=True)
class CaptureReference:
    take_id: str
    take_dir: Path
    metadata: dict
    modalities: list[CaptureModality]
    assets: dict[str, dict[str, str]]
    frame_count: int | None = None

    def has_modality(self, modality: CaptureModality) -> bool:
        return modality in self.modalities


@dataclass(frozen=True)
class CaptureFrameSet:
    frame_count: int | None = None
    synchronized: bool | None = None
    timestamp_source: str | None = None


class AcquisitionSource(Protocol):
    """Abstraction for any capture source (sensor, replay, file, synthetic)."""

    def latest(self) -> CaptureReference:
        ...

    def by_id(self, take_id: str) -> CaptureReference:
        ...


class FileSource(AcquisitionSource):
    """Reads capture folders from the local filesystem queue."""

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir
        self.incoming_dir = data_dir / "incoming"
        self.processed_dir = data_dir / "processed"

    def latest(self) -> CaptureReference:
        candidates: list[tuple[str, Path, dict]] = []
        if not self.incoming_dir.is_dir():
            raise FileNotFoundError(f"Incoming directory not found: {self.incoming_dir}")
        for take_dir in self.incoming_dir.iterdir():
            if not take_dir.is_dir() or not (take_dir / "READY").is_file():
                continue
            metadata = self._read_metadata(take_dir)
            created_at = str(metadata.get("created_at") or take_dir.name)
            candidates.append((created_at, take_dir, metadata))
        if not candidates:
            raise FileNotFoundError("No ready captures found.")
        candidates.sort(key=lambda item: item[0], reverse=True)
        _, take_dir, metadata = candidates[0]
        return self._reference(take_dir.name, take_dir, metadata)

    def by_id(self, take_id: str) -> CaptureReference:
        take_dir = self.incoming_dir / take_id
        if not take_dir.is_dir():
            raise FileNotFoundError(f"Take folder not found: {take_dir}")
        if not (take_dir / "READY").is_file():
            raise FileNotFoundError(f"Take is not ready: {take_dir}")
        metadata = self._read_metadata(take_dir)
        return self._reference(take_id, take_dir, metadata)

    @staticmethod
    def _read_metadata(take_dir: Path) -> dict:
        metadata_path = take_dir / "metadata.json"
        if not metadata_path.is_file():
            return {}
        return json.loads(metadata_path.read_text(encoding="utf-8"))

    @staticmethod
    def _reference(take_id: str, take_dir: Path, metadata: dict) -> CaptureReference:
        modalities = infer_modalities(metadata, take_dir)
        assets = assets_by_modality(metadata, take_dir)
        frameset = metadata.get("frameset") if isinstance(metadata.get("frameset"), dict) else {}
        frame_count = frameset.get("frame_count") or metadata.get("frame_count")
        return CaptureReference(
            take_id=take_id,
            take_dir=take_dir,
            metadata=metadata,
            modalities=modalities,
            assets=assets,
            frame_count=int(frame_count) if frame_count else None,
        )


class ReplaySource(FileSource):
    """Replay source backed by recorded capture folders."""


class SensorSource(AcquisitionSource):
    """Adapter placeholder for live device acquisition paths."""

    def latest(self) -> CaptureReference:
        raise NotImplementedError("SensorSource.latest is not implemented in this POC.")

    def by_id(self, take_id: str) -> CaptureReference:
        raise NotImplementedError("SensorSource.by_id is not implemented in this POC.")
