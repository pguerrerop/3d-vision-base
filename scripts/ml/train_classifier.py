#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from vision_3d_acquisition.ml import MLService


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--id", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--classifier-type", default="random_forest")
    parser.add_argument("--feature-set-id", default="core_25d_v1")
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--split-strategy", default="stratified")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    svc = MLService(Path(args.data_dir))
    out = svc.run_experiment(
        {
            "id": args.id,
            "name": args.name,
            "classifier_type": args.classifier_type,
            "feature_set_id": args.feature_set_id,
            "dataset_id": args.dataset_id,
            "split_strategy": args.split_strategy,
            "training_config": {"seed": args.seed},
        }
    )
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
