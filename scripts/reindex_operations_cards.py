#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from vision_3d_acquisition.operations.summary import reindex_recent_operations_cards


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recompute operations cards/index from existing result.json payloads.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--limit", type=int, default=200)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_dir = Path(args.data_dir).expanduser().resolve()
    summary = reindex_recent_operations_cards(data_dir, limit=max(1, int(args.limit)))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
