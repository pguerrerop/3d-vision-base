from pathlib import Path
import tempfile

import numpy as np

from vision_3d_acquisition.acquisition.base import AcquisitionBase
from vision_3d_acquisition.storage.publisher import AcquisitionPublisher
from vision_3d_acquisition.contracts.metadata import (
    AcquisitionMetadata,
    FileReferences,
)
from vision_3d_acquisition.processing.pointcloud_io import load_point_cloud, save_point_cloud_npz
from vision_3d_acquisition.utils.ids import generate_take_id
from vision_3d_acquisition.utils.time import utc_now_iso

POINT_CLOUD_NAME = "point_cloud.ply"
POINT_CLOUD_NPZ_NAME = "point_cloud.npz"


class OfflinePlyAcquisition(AcquisitionBase):
    """Ingest an existing PLY file and publish it as a take."""

    def __init__(self, publisher: AcquisitionPublisher) -> None:
        self.publisher = publisher

    def acquire(self, ply_path: Path, *, acquisition_group_id: str | None = None) -> tuple[str, Path]:
        src = self._validate_ply(ply_path)
        take_id = generate_take_id()
        created_at = utc_now_iso()

        metadata = AcquisitionMetadata(
            take_id=take_id,
            source="offline_ply",
            mode="offline",
            created_at=created_at,
            frame_count=1,
            acquisition_group_id=acquisition_group_id,
            files=FileReferences(point_cloud=POINT_CLOUD_NAME, point_cloud_npz=POINT_CLOUD_NPZ_NAME),
        )

        with tempfile.TemporaryDirectory(prefix="vision_3d_npz_") as temp_dir:
            npz_path = Path(temp_dir) / POINT_CLOUD_NPZ_NAME
            try:
                save_point_cloud_npz(load_point_cloud(src), npz_path)
            except ValueError:
                save_point_cloud_npz(np.empty((0, 3), dtype=np.float32), npz_path)
            folder = self.publisher.publish_take(
                take_id,
                metadata,
                {"point_cloud": src, "point_cloud_npz": npz_path},
            )
        return take_id, folder

    @staticmethod
    def _validate_ply(ply_path: Path) -> Path:
        path = ply_path.expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"PLY file not found: {path}")
        if not path.is_file():
            raise ValueError(f"Not a file: {path}")
        if path.suffix.lower() != ".ply":
            raise ValueError(f"Expected a .ply file, got: {path.name}")
        return path
