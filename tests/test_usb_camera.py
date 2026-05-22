from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from vision_3d_acquisition.acquisition.usb_camera import apply_camera_controls, camera_controls_snapshot, capture_image, discover_cameras, record_video
from vision_3d_acquisition.acquisition.preview import read_preview_metadata
from vision_3d_acquisition.contracts.metadata import AcquisitionMetadata
from vision_3d_acquisition.contracts.modalities import infer_modalities
from vision_3d_acquisition.state.runtime_status import read_runtime_status


class FakeCapture:
    def __init__(self, index: int, opened: bool = True) -> None:
        self.index = index
        self.opened = opened
        self.released = False
        self.frame = np.zeros((24, 32, 3), dtype=np.uint8)
        self.props = {3: 32, 4: 24, 5: 15}

    def isOpened(self) -> bool:
        return self.opened

    def read(self):
        return self.opened, self.frame.copy()

    def release(self) -> None:
        self.released = True

    def get(self, prop: int) -> float:
        return float(self.props.get(prop, 0))

    def set(self, prop: int, value: float) -> None:
        self.props[prop] = value

    def getBackendName(self) -> str:
        return "FAKE"


class FakeWriter:
    def __init__(self, path: str, opened: bool = True) -> None:
        self.path = Path(path)
        self.opened = opened
        self.frames = 0

    def isOpened(self) -> bool:
        return self.opened

    def write(self, frame) -> None:
        self.frames += 1
        self.path.write_bytes(b"fake-video")

    def release(self) -> None:
        if self.frames and not self.path.exists():
            self.path.write_bytes(b"fake-video")


class FakeCv2:
    CAP_PROP_FRAME_WIDTH = 3
    CAP_PROP_FRAME_HEIGHT = 4
    CAP_PROP_FPS = 5
    FONT_HERSHEY_SIMPLEX = 0
    IMWRITE_JPEG_QUALITY = 1
    LINE_AA = 16
    CAP_PROP_AUTO_EXPOSURE = 21
    CAP_PROP_EXPOSURE = 15
    CAP_PROP_AUTOFOCUS = 39
    CAP_PROP_FOCUS = 28
    CAP_PROP_GAIN = 14
    CAP_PROP_BRIGHTNESS = 10
    CAP_PROP_CONTRAST = 11
    CAP_PROP_SHARPNESS = 20
    CAP_PROP_SATURATION = 12
    CAP_PROP_WHITE_BALANCE_BLUE_U = 17

    def __init__(self, opened: dict[int, bool] | None = None) -> None:
        self.opened = opened or {0: True}

    def VideoCapture(self, index: int) -> FakeCapture:
        return FakeCapture(index, self.opened.get(index, False))

    def imwrite(self, path: str, frame, params=None) -> bool:
        Path(path).write_bytes(b"fake-image")
        return True

    def VideoWriter_fourcc(self, *codec: str) -> int:
        return 1

    def VideoWriter(self, path: str, fourcc: int, fps: float, size: tuple[int, int]) -> FakeWriter:
        return FakeWriter(path)

    def putText(self, *args, **kwargs) -> None:
        return None

    def imshow(self, *args, **kwargs) -> None:
        return None

    def waitKey(self, delay: int) -> int:
        return ord("q")

    def destroyAllWindows(self) -> None:
        return None


def test_discover_cameras_reports_working_and_unavailable() -> None:
    cameras = discover_cameras(max_index=2, cv2_module=FakeCv2({0: True, 1: False}))

    assert cameras[0].working is True
    assert cameras[0].resolution() == [32, 24]
    assert cameras[0].backend == "FAKE"
    assert cameras[1].working is False
    assert cameras[2].working is False


def test_capture_image_publishes_rgb_take_with_session_and_runtime_state(tmp_path: Path) -> None:
    take_id, folder = capture_image(
        data_dir=tmp_path,
        camera_index=0,
        session_id="session_rgb",
        output_name="rgb_take_001",
        cv2_module=FakeCv2({0: True}),
    )

    metadata = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
    validated = AcquisitionMetadata.model_validate(metadata)
    runtime = read_runtime_status(tmp_path / "state")

    assert take_id == "rgb_take_001"
    assert (folder / "READY").is_file()
    assert (folder / "rgb.png").is_file()
    assert validated.modalities == ["rgb"]
    assert validated.session_id == "session_rgb"
    assert metadata["source"]["type"] == "usb_camera"
    assert runtime["acquisition_source"] == "usb_camera"
    assert runtime["acquisition_source_details"]["camera_index"] == 0
    assert runtime["preview_available"] is True
    assert runtime["preview_stale"] is False
    assert (tmp_path / "runtime" / "previews" / "usb_camera_0.jpg").is_file()
    assert (tmp_path / "sessions" / "session_rgb" / "takes" / take_id / "metadata.json").is_file()


def test_record_video_publishes_rgb_video_take(tmp_path: Path) -> None:
    take_id, folder = record_video(
        data_dir=tmp_path,
        camera_index=0,
        duration=0.01,
        output_name="rgb_video_take_001",
        cv2_module=FakeCv2({0: True}),
    )

    metadata = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))

    assert take_id == "rgb_video_take_001"
    assert (folder / "preview.png").is_file()
    assert (folder / metadata["files"]["rgb_video"]).is_file()
    assert metadata["modalities"] == ["rgb", "rgb_video"]
    assert infer_modalities(metadata, folder) == ["rgb", "rgb_video"]
    assert metadata["frameset"]["frame_count"] >= 1
    assert metadata["source"]["duration_seconds"] > 0
    preview = read_preview_metadata(tmp_path)
    assert preview["source"] == "usb_camera"
    assert preview["camera_index"] == 0
    assert preview["resolution"] == [32, 24]
    assert preview["stale"] is False


def test_preview_metadata_reports_missing_and_stale(tmp_path: Path) -> None:
    missing = read_preview_metadata(tmp_path)
    assert missing["stale"] is True
    assert missing["error"] == "no preview available"

    preview_dir = tmp_path / "runtime" / "previews"
    preview_dir.mkdir(parents=True)
    (preview_dir / "usb_camera_0.json").write_text(
        json.dumps(
            {
                "source": "usb_camera",
                "camera_index": 0,
                "timestamp": "2000-01-01T00:00:00+00:00",
                "resolution": [32, 24],
                "stale": False,
            }
        ),
        encoding="utf-8",
    )

    stale = read_preview_metadata(tmp_path)
    assert stale["stale"] is True


def test_camera_controls_snapshot_and_apply_roundtrip() -> None:
    cv2 = FakeCv2({0: True})
    snap = camera_controls_snapshot(0, cv2_module=cv2)
    assert snap["backend"] == "FAKE"
    assert snap["controls"]["exposure"]["supported"] is True
    assert "readable" in snap["controls"]["exposure"]
    assert "writable" in snap["controls"]["exposure"]
    assert "backend_property" in snap["controls"]["exposure"]
    assert "range_source" in snap["controls"]["exposure"]

    applied = apply_camera_controls(0, {"exposure": {"value": -6, "auto": False}, "focus": {"value": 20, "auto": False}}, cv2_module=cv2)
    assert applied["controls"]["exposure"]["supported"] is True
    assert "applied" in applied


def test_unsupported_controls_handled_gracefully() -> None:
    class PartialCv2:
        CAP_PROP_EXPOSURE = 15
        CAP_PROP_FRAME_WIDTH = 3
        CAP_PROP_FRAME_HEIGHT = 4
        CAP_PROP_FPS = 5
        FONT_HERSHEY_SIMPLEX = 0
        IMWRITE_JPEG_QUALITY = 1
        LINE_AA = 16

        def __init__(self, opened: dict[int, bool] | None = None) -> None:
            self.opened = opened or {0: True}

        def VideoCapture(self, index: int) -> FakeCapture:
            return FakeCapture(index, self.opened.get(index, False))

    cv2 = PartialCv2({0: True})
    snap = camera_controls_snapshot(0, cv2_module=cv2)
    assert "gain" in snap["controls"]
    assert snap["controls"]["gain"]["supported"] is False
    assert snap["controls"]["gain"]["readable"] is False
    assert snap["controls"]["gain"]["writable"] is False


def test_apply_ignores_unsupported_controls_with_warning() -> None:
    cv2 = FakeCv2({0: True})
    applied = apply_camera_controls(0, {"unknown_control": {"value": 1}, "exposure": {"value": -5}}, cv2_module=cv2)
    assert applied["controls"]["exposure"]["supported"] is True
