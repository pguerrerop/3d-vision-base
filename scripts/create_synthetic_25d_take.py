#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from vision_3d_acquisition.synthetic import create_synthetic_25d_take  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Create synthetic offline 25D heightmap take(s).")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Data root")
    parser.add_argument("--session-id", type=str, default="synthetic_25d_demo", help="Session id")
    parser.add_argument("--count", type=int, default=1, help="Number of takes to create")
    parser.add_argument("--no-reflectance", action="store_true", help="Disable reflectance output")
    args = parser.parse_args()

    created: list[tuple[str, Path]] = []
    for i in range(max(1, args.count)):
        take_id, take_dir = create_synthetic_25d_take(
            args.data_dir,
            session_id=args.session_id,
            include_reflectance=not args.no_reflectance,
            seed=25 + i,
        )
        created.append((take_id, take_dir))

    for take_id, take_dir in created:
        print(f"take_id: {take_id}")
        print(f"take_dir: {take_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
