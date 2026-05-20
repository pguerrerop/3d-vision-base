import json
from pathlib import Path

import pytest

from vision_3d_acquisition.acquisition.offline_ply import OfflinePlyAcquisition
from vision_3d_acquisition.acquisition.publisher import AcquisitionPublisher as LegacyPublisher
from vision_3d_acquisition.contracts.metadata import AcquisitionMetadata
from vision_3d_acquisition.storage.publisher import AcquisitionPublisher

REPO_ROOT = Path(__file__).resolve().parents[1]
TRACKED_SAMPLE_PLY = REPO_ROOT / "samples" / "pointclouds" / "sample.ply"


@pytest.fixture
def sample_ply(tmp_path: Path) -> Path:
    ply = tmp_path / "sample.ply"
    ply.write_text("ply\nformat ascii 1.0\nend_header\n", encoding="utf-8")
    return ply


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    return tmp_path / "data"


def test_publish_ply_take(sample_ply: Path, data_dir: Path) -> None:
    publisher = AcquisitionPublisher(data_dir)
    take_id, folder = OfflinePlyAcquisition(publisher).acquire(sample_ply)

    assert folder == data_dir / "incoming" / take_id
    assert (folder / "READY").is_file()
    assert (folder / "point_cloud.ply").is_file()
    assert (folder / "point_cloud.npz").is_file()

    meta = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
    validated = AcquisitionMetadata.model_validate(meta)
    assert validated.take_id == take_id
    assert validated.source == "offline_ply"
    assert validated.mode == "offline"
    assert validated.files.point_cloud == "point_cloud.ply"
    assert validated.files.point_cloud_npz == "point_cloud.npz"
    assert validated.modalities == ["point_cloud"]

    state = json.loads(
        (data_dir / "state" / "acquisition.json").read_text(encoding="utf-8")
    )
    assert state["last_take_id"] == take_id


def test_publish_tracked_sample_ply(data_dir: Path) -> None:
    if not TRACKED_SAMPLE_PLY.is_file():
        pytest.skip(f"tracked sample not found: {TRACKED_SAMPLE_PLY}")

    publisher = AcquisitionPublisher(data_dir)
    take_id, folder = OfflinePlyAcquisition(publisher).acquire(TRACKED_SAMPLE_PLY)

    assert (folder / "READY").is_file()
    assert (folder / "point_cloud.ply").is_file()
    assert (folder / "point_cloud.npz").is_file()
    meta = AcquisitionMetadata.model_validate_json(
        (folder / "metadata.json").read_text(encoding="utf-8")
    )
    assert meta.take_id == take_id


def test_backward_compatible_publisher_import(data_dir: Path, sample_ply: Path) -> None:
    assert LegacyPublisher is AcquisitionPublisher
    publisher = LegacyPublisher(data_dir)
    take_id, folder = OfflinePlyAcquisition(publisher).acquire(sample_ply)
    assert (folder / "READY").exists()


def test_missing_ply(data_dir: Path) -> None:
    publisher = AcquisitionPublisher(data_dir)
    with pytest.raises(FileNotFoundError):
        OfflinePlyAcquisition(publisher).acquire(Path("/no/such/file.ply"))


def test_invalid_extension(data_dir: Path, tmp_path: Path) -> None:
    txt = tmp_path / "not_ply.txt"
    txt.write_text("x", encoding="utf-8")
    publisher = AcquisitionPublisher(data_dir)
    with pytest.raises(ValueError, match=r"\.ply"):
        OfflinePlyAcquisition(publisher).acquire(txt)
