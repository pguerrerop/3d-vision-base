#!/usr/bin/env python3
"""Publish an offline PLY file as a take on the filesystem queue."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from vision_3d_acquisition.acquisition.offline_ply import OfflinePlyAcquisition
from vision_3d_acquisition.storage.publisher import AcquisitionPublisher


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Publish a .ply file as an acquisition take.",
    )
    parser.add_argument(
        "--ply",
        required=True,
        type=Path,
        help="Path to the source .ply file",
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        type=Path,
        help="Data root (incoming/, processed/, state/)",
    )
    args = parser.parse_args()

    try:
        publisher = AcquisitionPublisher(args.data_dir)
        acquisition = OfflinePlyAcquisition(publisher)
        take_id, folder = acquisition.acquire(args.ply)
    except (FileNotFoundError, ValueError, FileExistsError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"take_id: {take_id}")
    metadata = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
    print(f"modalities: {', '.join(metadata.get('modalities') or ['point_cloud'])}")
    print(f"folder: {folder.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
