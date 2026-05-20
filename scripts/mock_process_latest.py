#!/usr/bin/env python3
"""Create mock processed output for the latest ready incoming take."""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import UTC, datetime
from pathlib import Path

from PIL import Image, ImageDraw

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from vision_3d_acquisition.contracts.result import (  # noqa: E402
    DetectedObject,
    ProcessingResult,
    ProcessingTiming,
    ResultFiles,
    ResultSummary,
)
from vision_3d_acquisition.processing.pointcloud_preview import (  # noqa: E402
    compute_point_cloud_stats,
    render_point_cloud_preview,
)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def latest_unprocessed_take(data_dir: Path) -> Path | None:
    incoming = data_dir / "incoming"
    processed = data_dir / "processed"
    if not incoming.exists():
        return None
    candidates: list[tuple[str, Path]] = []
    for take_dir in incoming.iterdir():
        if not take_dir.is_dir() or not (take_dir / "READY").is_file():
            continue
        if (processed / take_dir.name / "DONE").exists():
            continue
        metadata = read_json(take_dir / "metadata.json") if (take_dir / "metadata.json").is_file() else {}
        candidates.append((metadata.get("created_at") or take_dir.name, take_dir))
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1] if candidates else None


def create_placeholder_png(path: Path, title: str, accent: tuple[int, int, int], seed: int) -> None:
    rng = random.Random(seed)
    width, height = 960, 540
    image = Image.new("RGB", (width, height), (22, 28, 36))
    draw = ImageDraw.Draw(image)
    for y in range(0, height, 18):
        shade = 30 + int(y / height * 50)
        draw.rectangle((0, y, width, y + 18), fill=(shade, shade + 6, shade + 12))
    for _ in range(22):
        x = rng.randint(80, width - 120)
        y = rng.randint(90, height - 110)
        r = rng.randint(12, 36)
        color = tuple(min(255, max(0, channel + rng.randint(-25, 25))) for channel in accent)
        draw.ellipse((x - r, y - r, x + r, y + r), outline=color, width=4)
    draw.rectangle((32, 32, width - 32, 92), fill=(245, 247, 250))
    draw.text((56, 54), title, fill=(18, 24, 32))
    image.save(path)


def build_result(take_id: str, metadata: dict, input_stats: dict) -> ProcessingResult:
    seed = sum(ord(char) for char in take_id)
    rng = random.Random(seed)
    ball_count = rng.randint(1, 4)
    non_ball_count = rng.randint(0, 2)
    decision = "accept" if non_ball_count == 0 else "review" if ball_count >= non_ball_count else "reject"
    confidence = round(rng.uniform(0.78, 0.96), 3)
    objects: list[DetectedObject] = []
    object_id = 1
    for _ in range(ball_count):
        diameter = round(rng.uniform(22.0, 29.0), 2)
        objects.append(
            DetectedObject(
                object_id=object_id,
                class_name="ball",
                confidence=round(rng.uniform(0.82, 0.98), 3),
                center_mm=(round(rng.uniform(-60, 60), 2), round(rng.uniform(-40, 40), 2), round(rng.uniform(8, 35), 2)),
                diameter_mm=diameter,
                sphericity_score=round(rng.uniform(0.86, 0.99), 3),
                fit_rmse_mm=round(rng.uniform(0.12, 0.55), 3),
            )
        )
        object_id += 1
    for _ in range(non_ball_count):
        objects.append(
            DetectedObject(
                object_id=object_id,
                class_name="non_ball",
                confidence=round(rng.uniform(0.63, 0.86), 3),
                center_mm=(round(rng.uniform(-70, 70), 2), round(rng.uniform(-50, 50), 2), round(rng.uniform(6, 32), 2)),
                diameter_mm=None,
                sphericity_score=round(rng.uniform(0.25, 0.68), 3),
                fit_rmse_mm=round(rng.uniform(0.7, 1.8), 3),
            )
        )
        object_id += 1
    load = rng.randint(35, 80)
    segmentation = rng.randint(90, 180)
    classification = rng.randint(25, 70)
    total = load + segmentation + classification + rng.randint(8, 20)
    return ProcessingResult(
        take_id=take_id,
        session_id=metadata.get("session_id"),
        frameset_id=((metadata.get("frameset") or {}).get("frameset_id") if isinstance(metadata.get("frameset"), dict) else None),
        processed_at=datetime.now(UTC),
        processing_mode="mock",
        algorithm_stage="mock",
        status="ok",
        summary=ResultSummary(
            object_count=len(objects),
            ball_count=ball_count,
            non_ball_count=non_ball_count,
            decision=decision,
            confidence=confidence,
        ),
        input_stats=input_stats,
        objects=objects,
        files=ResultFiles(
            point_cloud=metadata.get("files", {}).get("point_cloud", "point_cloud.ply"),
            input_preview="input_point_cloud_preview.png",
            overlay="overlay.png",
            debug_height="debug_height.png",
            debug_segmentation="debug_segmentation.png",
        ),
        timing_ms=ProcessingTiming(
            load=load,
            segmentation=segmentation,
            classification=classification,
            total=total,
        ),
        error=None,
        synchronization=((metadata.get("frameset") or {}).get("synchronization") if isinstance(metadata.get("frameset"), dict) else None),
        acquisition_timestamps={"acquired_at": metadata.get("created_at"), "metadata_created_at": metadata.get("created_at")},
    )


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Mock-process the latest ready incoming take.")
    parser.add_argument("--data-dir", default="data", type=Path, help="Data root")
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    for child in ("incoming", "processed", "state", "events"):
        (data_dir / child).mkdir(parents=True, exist_ok=True)

    take_dir = latest_unprocessed_take(data_dir)
    if take_dir is None:
        print("No ready unprocessed take found.", file=sys.stderr)
        return 1

    take_id = take_dir.name
    metadata = read_json(take_dir / "metadata.json")
    output_dir = data_dir / "processed" / take_id
    output_dir.mkdir(parents=True, exist_ok=True)
    point_cloud_filename = metadata.get("files", {}).get("point_cloud", "point_cloud.ply")
    point_cloud_path = take_dir / point_cloud_filename

    input_preview = output_dir / "input_point_cloud_preview.png"
    input_stats = compute_point_cloud_stats(point_cloud_path)
    render_point_cloud_preview(point_cloud_path, input_preview)

    create_placeholder_png(output_dir / "overlay.png", f"Overlay preview: {take_id}", (73, 190, 138), seed=1)
    create_placeholder_png(output_dir / "debug_height.png", f"Height map: {take_id}", (88, 166, 255), seed=2)
    create_placeholder_png(output_dir / "debug_segmentation.png", f"Segmentation: {take_id}", (244, 180, 76), seed=3)

    result = build_result(take_id, metadata, input_stats)
    result_payload = result.model_dump(mode="json")
    write_json(output_dir / "result.json", result_payload)
    (output_dir / "DONE").touch()

    latest_payload = {
        "take_id": take_id,
        "session_id": metadata.get("session_id"),
        "frameset_id": ((metadata.get("frameset") or {}).get("frameset_id") if isinstance(metadata.get("frameset"), dict) else None),
        "processed_at": result_payload["processed_at"],
        "processing_mode": result_payload["processing_mode"],
        "algorithm_stage": result_payload["algorithm_stage"],
        "status": result.status,
        "summary": result_payload["summary"],
        "input_stats": result_payload["input_stats"],
        "files": result_payload["files"],
        "timing_ms": result_payload["timing_ms"],
        "runtime_acquisition": result_payload.get("runtime_acquisition"),
        "throughput": result_payload.get("throughput"),
    }
    write_json(data_dir / "state" / "latest.json", latest_payload)
    write_json(
        data_dir / "state" / "runtime.json",
        {
            "status": "ready",
            "latest_take_id": take_id,
            "latest_processed_take_id": take_id,
            "message": "Mock processing complete; detections are demonstration data.",
            "updated_at": result_payload["processed_at"],
            "acquisition_connected": True,
            "latest_frame_timestamp": metadata.get("created_at"),
            "processing_lag_ms": 0.0,
            "queue_size": 0,
            "dropped_frames": 0,
            "last_successful_processing": result_payload["processed_at"],
            "current_session": metadata.get("session_id"),
            "active_calibration": None,
            "acquisition_source": metadata.get("source"),
            "stale": False,
            "throughput": {},
            "warnings": [],
        },
    )
    event = {
        "event": "processed",
        "take_id": take_id,
        "created_at": result_payload["processed_at"],
        "processing_mode": result_payload["processing_mode"],
        "algorithm_stage": result_payload["algorithm_stage"],
        "summary": result_payload["summary"],
    }
    with (data_dir / "events" / "events.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event) + "\n")

    print(f"processed_take: {take_id}")
    print(f"folder: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
