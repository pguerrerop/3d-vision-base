from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from scripts.generate_rgb_steel_ball_samples import OUT_DIR, generate


def test_demo_sample_generator_creates_deterministic_rgb_set(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("scripts.generate_rgb_steel_ball_samples.OUT_DIR", tmp_path / "rgb_steel_balls")
    metadata = generate()
    out_dir: Path = tmp_path / "rgb_steel_balls"
    expected = {
        "one_good_ball.png",
        "multiple_good_balls.png",
        "oval_non_spherical.png",
        "mixed_objects.png",
        "low_contrast_ball.png",
    }
    assert expected == set(metadata.keys())

    for filename in expected:
        path = out_dir / filename
        assert path.is_file()
        image = Image.open(path)
        assert image.mode == "RGB"
        assert image.format == "PNG"

    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    loaded = json.loads((out_dir / "metadata.json").read_text(encoding="utf-8"))
    assert loaded["one_good_ball.png"]["expected_object_count"] == 1
