#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from vision_3d_acquisition.ml import MLService


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--notes", default=None)
    args = parser.parse_args()
    svc = MLService(Path(args.data_dir))
    print(svc.promote_model(model_id=args.model_id, notes=args.notes))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
