#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from vision_3d_acquisition.acquisition.trispector_25d import parse_trispector_2_5d_image  # noqa: E402
from vision_3d_acquisition.runtime.process_registry import ProcessHeartbeat, RuntimeProcessRegistry  # noqa: E402
from vision_3d_acquisition.utils.ids import generate_take_id  # noqa: E402


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Watch TriSpector FTP folder and register acquired 2.5D takes.")
    parser.add_argument("--watch-dir", required=True)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--source-id", default="trispector_ftp")
    parser.add_argument("--poll-interval-sec", type=float, default=1.0)
    parser.add_argument("--stability-checks", type=int, default=3)
    parser.add_argument("--stability-interval-sec", type=float, default=0.75)
    parser.add_argument("--done-marker-suffix", default=".done")
    parser.add_argument("--claim-suffix", default=".claim")
    parser.add_argument("--processed-suffix", default=".processed")
    parser.add_argument("--failed-suffix", default=".failed")
    parser.add_argument("--once", action="store_true")
    return parser.parse_args()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def is_capture_complete(path: Path, *, checks: int, wait_sec: float, done_marker_suffix: str) -> bool:
    if (path.with_name(path.name + done_marker_suffix)).is_file():
        return True
    previous = -1
    stable = 0
    for _ in range(max(1, checks)):
        try:
            size = path.stat().st_size
        except FileNotFoundError:
            return False
        if size > 0 and size == previous:
            stable += 1
        else:
            stable = 0
        previous = size
        if stable >= max(1, checks - 1):
            return True
        time.sleep(max(0.1, wait_sec))
    return False


def build_take_metadata(take_id: str, source_id: str, published_files: dict[str, str], uploaded_filename: str) -> dict[str, Any]:
    created_at = now_iso()
    return {
        "take_id": take_id,
        "source": {
            "type": "ftp",
            "sensor": "trispector_2_5d",
            "source_id": source_id,
            "uploaded_filename": uploaded_filename,
        },
        "mode": "live",
        "created_at": created_at,
        "frame_count": 1,
        "modalities": ["heightmap"],
        "files": published_files,
        "frameset": {
            "frameset_id": f"{take_id}_fs0",
            "timestamp": created_at,
            "frame_count": 1,
            "synchronized": False,
            "timestamp_source": "ftp_upload",
        },
    }


def register_take(data_dir: Path, source_id: str, upload_file: Path) -> str:
    take_id = generate_take_id()
    incoming_dir = data_dir / "incoming" / take_id
    incoming_dir.mkdir(parents=True, exist_ok=False)

    with tempfile.TemporaryDirectory(prefix="trispector_watch_") as tmp_raw:
        tmp_dir = Path(tmp_raw)
        raw_name = f"raw_upload{upload_file.suffix.lower() or '.png'}"
        raw_copy = tmp_dir / raw_name
        raw_copy.write_bytes(upload_file.read_bytes())
        parsed = parse_trispector_2_5d_image(raw_copy, tmp_dir)

        files_to_copy = {
            "heightmap": parsed["heightmap"],
            "reflectance": parsed["reflectance"],
            "raw_upload": raw_copy,
            "parser_metadata": parsed["parser_metadata"],
            "heightmap_preview": parsed["heightmap_preview"],
        }
        published_files: dict[str, str] = {
            "heightmap": "height16.tif",
            "reflectance": "reflectance.png",
            "raw_upload": raw_name,
            "parser_metadata": "parser_metadata.json",
            "heightmap_preview": "heightmap_preview.png",
        }
        for _, src in files_to_copy.items():
            dst = incoming_dir / src.name
            shutil.copy2(src, dst)

    metadata = build_take_metadata(take_id, source_id, published_files, upload_file.name)
    write_json(incoming_dir / "metadata.json", metadata)
    (incoming_dir / "READY").touch()
    write_json(
        incoming_dir / "runtime_state.json",
        {
            "take_id": take_id,
            "state": "acquired",
            "created_at": now_iso(),
            "updated_at": now_iso(),
            "source": "trispector_ftp",
            "modalities": ["heightmap"],
        },
    )
    return take_id


def iter_uploads(watch_dir: Path) -> list[Path]:
    if not watch_dir.is_dir():
        return []
    files = [
        p
        for p in watch_dir.iterdir()
        if p.is_file() and not p.name.startswith(".") and p.suffix.lower() in {".png", ".tif", ".tiff", ".bmp"}
    ]
    files.sort(key=lambda p: p.stat().st_mtime)
    return files


def main() -> int:
    args = parse_args()
    watch_dir = Path(args.watch_dir).expanduser().resolve()
    data_dir = Path(args.data_dir).expanduser().resolve()
    watch_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "incoming").mkdir(parents=True, exist_ok=True)
    heartbeat = ProcessHeartbeat(
        RuntimeProcessRegistry(data_dir),
        process_name="trispector_take_watcher",
        runtime_role="watcher",
        pid=os.getpid(),
        interval_seconds=3.0,
        base_metadata={"watch_dir": str(watch_dir)},
    )
    heartbeat.start()

    acquired_total = 0
    errors_total = 0
    try:
        while True:
            processed_in_loop = 0
            for upload_file in iter_uploads(watch_dir):
                claim = upload_file.with_name(upload_file.name + args.claim_suffix)
                processed = upload_file.with_name(upload_file.name + args.processed_suffix)
                failed = upload_file.with_name(upload_file.name + args.failed_suffix)
                if processed.is_file() or failed.is_file() or claim.is_file():
                    continue
                if not is_capture_complete(
                    upload_file,
                    checks=max(1, int(args.stability_checks)),
                    wait_sec=max(0.1, float(args.stability_interval_sec)),
                    done_marker_suffix=args.done_marker_suffix,
                ):
                    continue

                try:
                    claim.write_text(now_iso(), encoding="utf-8")
                    take_id = register_take(data_dir, args.source_id, upload_file)
                    processed.write_text(json.dumps({"take_id": take_id, "processed_at": now_iso()}, indent=2) + "\n", encoding="utf-8")
                    print(json.dumps({"status": "acquired", "take_id": take_id, "file": str(upload_file.name)}))
                    processed_in_loop += 1
                    acquired_total += 1
                except Exception as exc:  # noqa: BLE001
                    failed.write_text(json.dumps({"error": str(exc), "failed_at": now_iso()}, indent=2) + "\n", encoding="utf-8")
                    print(json.dumps({"status": "failed", "file": str(upload_file.name), "error": str(exc)}))
                    errors_total += 1
                finally:
                    claim.unlink(missing_ok=True)

            heartbeat.pulse(counters={"acquired_total": acquired_total, "errors_total": errors_total})
            if args.once:
                break
            if processed_in_loop == 0:
                time.sleep(max(0.1, float(args.poll_interval_sec)))
    except Exception as exc:  # noqa: BLE001
        heartbeat.set_failed(str(exc))
        raise
    finally:
        heartbeat.stop(status="stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
