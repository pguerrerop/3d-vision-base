#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from vision_3d_acquisition.api.settings import ApiSettings
from vision_3d_acquisition.acquisition.replay_dataset import ReplayableAcquisitionService


def _settings() -> ApiSettings:
    data_dir = Path("data")
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
    parser = argparse.ArgumentParser(description="Replay a single take via normal pipeline execution contract.")
    parser.add_argument("--take-id", required=True)
    parser.add_argument("--pipeline", default=None)
    args = parser.parse_args()

    settings = _settings()
    payload = ReplayableAcquisitionService(settings.data_dir).replay_take(
        take_id=str(args.take_id),
        pipeline_id=str(args.pipeline) if args.pipeline else None,
    )
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
