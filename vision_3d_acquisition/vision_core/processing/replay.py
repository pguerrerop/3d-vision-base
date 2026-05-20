from __future__ import annotations

from pathlib import Path

from vision_3d_acquisition.vision_core.acquisition.sources import FileSource


def load_latest_capture(data_dir: Path) -> tuple[str, Path, dict]:
    source = FileSource(data_dir)
    reference = source.latest()
    return reference.take_id, reference.take_dir, reference.metadata
