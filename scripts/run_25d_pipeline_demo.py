#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from vision_3d_acquisition.apps.ball_inspection_25d import run_ball_inspection_25d_flow  # noqa: E402
from vision_3d_acquisition.synthetic import create_synthetic_25d_take  # noqa: E402


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Run native 25D pipeline demo on synthetic or existing take.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Data root")
    parser.add_argument("--take-id", type=str, default=None, help="Existing take id to process")
    parser.add_argument("--session-id", type=str, default="synthetic_25d_demo", help="Synthetic session id")
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    take_id = args.take_id
    if not take_id:
        take_id, take_dir = create_synthetic_25d_take(data_dir, session_id=args.session_id)
        print(f"created_take_id: {take_id}")
        print(f"created_take_dir: {take_dir}")

    result = run_ball_inspection_25d_flow(data_dir, take_id=take_id)
    result_path = result.output_dir / "result.json"
    payload = _read_json(result_path)

    print(f"pipeline_id: {((payload.get('processing_pipeline') or {}).get('id') or '-')}")
    print(f"pipeline_family: {((payload.get('processing_pipeline') or {}).get('pipeline_family') or '-')}")
    print(f"result_path: {result_path}")
    print(f"status: {payload.get('status')}")
    print(f"object_count: {(payload.get('summary') or {}).get('object_count')}")
    belt_plane_artifact = next((a for a in (payload.get("artifacts") or []) if a.get("artifact_id") == "belt_plane"), {})
    print(f"plane_residual_stats: {(belt_plane_artifact.get('metadata') or {}).get('residual_stats_mm')}")

    print("objects:")
    for item in payload.get("objects") or []:
        heights = item.get("height_above_belt_mm") or {}
        print(
            f"  - id={item.get('object_id')} class={item.get('class_name')} label={item.get('label')} "
            f"max_h={heights.get('max_height_mm')} mean_h={heights.get('mean_height_mm')} "
            f"p95_h={heights.get('p95_height_mm')} volume_proxy={item.get('feature_volume_proxy_mm3')}"
        )

    print("artifacts:")
    for artifact in payload.get("artifacts") or []:
        artifact_id = artifact.get("artifact_id")
        path = artifact.get("path")
        if path:
            print(f"  - {artifact_id}: {result.output_dir / path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
