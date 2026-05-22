#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from vision_3d_acquisition.api.main import ExecuteTakeRequest, process_take_for_pipeline  # noqa: E402
from vision_3d_acquisition.api.settings import ApiSettings  # noqa: E402
from vision_3d_acquisition.synthetic import create_synthetic_25d_take  # noqa: E402


def _settings(data_dir: Path) -> ApiSettings:
    settings = ApiSettings(
        data_dir=data_dir,
        incoming_dir=data_dir / "incoming",
        processed_dir=data_dir / "processed",
        state_dir=data_dir / "state",
        events_dir=data_dir / "events",
        sessions_dir=data_dir / "sessions",
        datasets_dir=data_dir / "datasets",
    )
    settings.ensure_directories()
    return settings


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate API processing contract for 25D pipeline.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--take-id", type=str, default=None)
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    take_id = args.take_id
    if not take_id:
        take_id, _ = create_synthetic_25d_take(data_dir)
        print(f"created_take_id: {take_id}")

    payload = ExecuteTakeRequest(pipeline_id="mining_steel_ball_classification_25d", reprocess=True)
    response = process_take_for_pipeline(take_id, payload, _settings(data_dir))
    print(json.dumps(response, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
