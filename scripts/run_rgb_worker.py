#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from vision_3d_acquisition.api.settings import ApiSettings  # noqa: E402
from vision_3d_acquisition.runtime.worker_manager import RuntimeWorkerManager  # noqa: E402
from vision_3d_acquisition.runtime.workers import RgbAcquisitionProcessingWorker  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RGB acquisition-processing runtime worker.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--source-id", default="usb_camera_0")
    parser.add_argument("--station-id", default=None)
    parser.add_argument("--poll-interval-sec", type=float, default=1.0)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _settings(data_dir: str) -> ApiSettings:
    root = Path(data_dir).expanduser().resolve()
    settings = ApiSettings(
        data_dir=root,
        incoming_dir=root / "incoming",
        processed_dir=root / "processed",
        state_dir=root / "state",
        events_dir=root / "events",
        sessions_dir=root / "sessions",
        datasets_dir=root / "datasets",
    )
    settings.ensure_directories()
    return settings


def main() -> int:
    args = parse_args()
    settings = _settings(args.data_dir)
    manager = RuntimeWorkerManager(settings)
    worker = RgbAcquisitionProcessingWorker(
        settings=settings,
        manager=manager,
        worker_id="rgb_worker",
        source_id=args.source_id,
        station_id=args.station_id,
        poll_interval_sec=args.poll_interval_sec,
    )
    summary = worker.run(once=bool(args.once), dry_run=bool(args.dry_run))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
