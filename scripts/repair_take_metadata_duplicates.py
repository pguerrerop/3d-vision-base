#!/usr/bin/env python3
"""One-shot repair for takes carrying a metadata sidecar under two sessions.

Background
----------
``/api/takes/bulk-metadata`` with ``action=move_to_session`` used to call
``upsert_take_metadata`` with the destination session id. That seeded the new
sidecar from ``default_take_metadata`` (the destination path did not exist yet)
and never removed the origin, so every moved take ended up with two sidecars:
the origin holding the real labels, the destination holding defaults. Which one
the API shows depends on ``Path.glob`` iteration order.

What this does
--------------
Completes the move that was attempted: the destination session keeps the take,
and the content comes from the origin sidecar, so labels, validation status and
physical object survive. The origin sidecar is then removed, and any session
left without takes is removed too.

The origin is the sidecar under the session with the earlier ``created_at``; the
destination is the later one. The script asserts that the origin is never poorer
than the destination before touching anything, so an unexpected shape aborts the
run rather than overwriting good data with defaults.

Usage
-----
    python scripts/repair_take_metadata_duplicates.py --data-dir data
    python scripts/repair_take_metadata_duplicates.py --data-dir data --apply
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tarfile
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Fields that carry operator work. A sidecar holding more of them is the one
# whose content has to survive the merge.
LIST_FIELDS = ("semantic_labels", "superclass_labels", "tags", "categories", "labels")
SCALAR_FIELDS = ("expected_class", "physical_object_id", "normalization_version", "normalized_class")
# Written as null by older code, as [] / False by newer code. Normalized on merge.
DEFAULTED_FIELDS = {"categories": list, "is_reference": bool, "is_golden_sample": bool}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def richness(payload: dict[str, Any]) -> int:
    """How much operator work a sidecar holds."""
    score = sum(len(payload.get(key) or []) for key in LIST_FIELDS if isinstance(payload.get(key), list))
    score += sum(1 for key in SCALAR_FIELDS if payload.get(key))
    score += 1 if str(payload.get("validation_status") or "unreviewed") != "unreviewed" else 0
    return score


def collect_sidecars(datasets_dir: Path) -> dict[str, list[tuple[str, str, Path]]]:
    sidecars: dict[str, list[tuple[str, str, Path]]] = defaultdict(list)
    for path in datasets_dir.glob("dataset_*/sessions/session_*/takes/*/metadata.json"):
        dataset_id = path.parts[-6].replace("dataset_", "", 1)
        session_id = path.parts[-4].replace("session_", "", 1)
        sidecars[path.parent.name].append((dataset_id, session_id, path))
    return sidecars


def session_created_at(datasets_dir: Path, dataset_id: str, session_id: str) -> str:
    path = datasets_dir / f"dataset_{dataset_id}" / "sessions" / f"session_{session_id}" / "session.json"
    try:
        return str(read_json(path).get("created_at") or "")
    except (OSError, json.JSONDecodeError):
        return ""


def normalize_defaults(payload: dict[str, Any]) -> dict[str, Any]:
    merged = dict(payload)
    for key, kind in DEFAULTED_FIELDS.items():
        if merged.get(key) is None:
            merged[key] = [] if kind is list else False
    return merged


def plan_repairs(data_dir: Path) -> tuple[list[dict[str, Any]], list[str]]:
    datasets_dir = data_dir / "datasets"
    problems: list[str] = []
    plan: list[dict[str, Any]] = []

    for take_id, entries in sorted(collect_sidecars(datasets_dir).items()):
        if len(entries) < 2:
            continue
        if len(entries) > 2:
            problems.append(f"{take_id}: {len(entries)} sidecars, expected 2")
            continue

        ordered = sorted(entries, key=lambda item: session_created_at(datasets_dir, item[0], item[1]))
        (origin_dataset, origin_session, origin_path), (dest_dataset, dest_session, dest_path) = ordered

        if origin_dataset != dest_dataset:
            problems.append(f"{take_id}: sidecars span datasets {origin_dataset} and {dest_dataset}")
            continue

        origin = read_json(origin_path)
        destination = read_json(dest_path)
        if richness(origin) < richness(destination):
            problems.append(
                f"{take_id}: destination session {dest_session} is richer than origin {origin_session}"
            )
            continue

        merged = normalize_defaults({**origin, "session_id": dest_session, "dataset_id": dest_dataset})
        plan.append(
            {
                "take_id": take_id,
                "dataset_id": dest_dataset,
                "origin_session": origin_session,
                "destination_session": dest_session,
                "origin_path": origin_path,
                "destination_path": dest_path,
                "labels_recovered": richness(origin) - richness(destination),
                "payload": merged,
                "changes_destination": merged != normalize_defaults(destination),
            }
        )

    return plan, problems


def empty_sessions(datasets_dir: Path) -> list[Path]:
    empty: list[Path] = []
    for session_dir in datasets_dir.glob("dataset_*/sessions/session_*"):
        takes_dir = session_dir / "takes"
        if not takes_dir.is_dir() or not any(takes_dir.iterdir()):
            empty.append(session_dir)
    return empty


def backup(data_dir: Path) -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    archive = data_dir / f"datasets_backup_{stamp}.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(data_dir / "datasets", arcname="datasets")
    return archive


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Data root (default: data)")
    parser.add_argument("--apply", action="store_true", help="Write the changes (default is a dry run)")
    parser.add_argument("--json", action="store_true", help="Emit the full plan as JSON")
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    plan, problems = plan_repairs(data_dir)

    if args.json:
        print(
            json.dumps(
                [{k: str(v) for k, v in row.items() if k != "payload"} for row in plan], indent=2
            )
        )

    moves = Counter((row["origin_session"], row["destination_session"]) for row in plan)
    recovered = sum(1 for row in plan if row["labels_recovered"] > 0)

    print(f"{'APPLY' if args.apply else 'DRY RUN'}  ·  {data_dir}")
    print(f"  takes con sidecar duplicado : {len(plan)}")
    print(f"  con etiquetas a recuperar   : {recovered}")
    print(f"  sin diferencia de contenido : {len(plan) - recovered}")
    print("  movimientos:")
    for (origin, destination), count in sorted(moves.items(), key=lambda item: -item[1]):
        print(f"    {count:4d}  {origin}  ->  {destination}")

    if problems:
        print(f"\n  ABORTADO: {len(problems)} casos no coinciden con el patron esperado:")
        for problem in problems[:10]:
            print(f"    {problem}")
        return 2

    if not args.apply:
        print("\n  Nada escrito. Volver a correr con --apply.")
        return 0

    archive = backup(data_dir)
    print(f"\n  backup: {archive}")

    for row in plan:
        write_json(row["destination_path"], row["payload"])
        shutil.rmtree(row["origin_path"].parent)

    removed_sessions = []
    for session_dir in empty_sessions(data_dir / "datasets"):
        shutil.rmtree(session_dir)
        removed_sessions.append(session_dir.name.replace("session_", "", 1))

    print(f"  sidecars fusionados: {len(plan)}")
    print(f"  sesiones vacias eliminadas: {len(removed_sessions)} {removed_sessions}")
    print("\n  Siguiente paso: python scripts/sensor_studio_cli.py index rebuild --data-dir data --full")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
