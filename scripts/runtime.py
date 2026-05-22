#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from vision_3d_acquisition.runtime.supervisor import get_runtime_supervisor  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Runtime process supervisor CLI.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list", help="List runtime processes.")

    for command in ("start", "stop", "restart", "logs", "selftest"):
        cmd = subparsers.add_parser(command, help=f"{command.title()} a runtime process.")
        cmd.add_argument("process_id", help="Runtime process id (example: trispector_ftp).")
        if command == "start":
            cmd.add_argument("--foreground", action="store_true", help="Run attached to terminal for live debugging.")
        if command == "logs":
            cmd.add_argument("--limit", type=int, default=100)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    supervisor = get_runtime_supervisor(register_atexit=False)
    if args.command == "list":
        processes = supervisor.list_processes()
        for item in processes:
            observed = str(item.get("observed_state") or "")
            desired = str(item.get("desired_state") or "")
            recovered = observed == "running" and desired == "running" and bool(item.get("pid"))
            item["status_rendered"] = f"{item.get('status')} (pid={item.get('pid') or '-'})"
            item["recovered_or_running"] = recovered
        payload = {"processes": processes}
    elif args.command == "start":
        if bool(getattr(args, "foreground", False)):
            command = supervisor.build_start_command(args.process_id)
            proc = subprocess.Popen(command)
            try:
                return proc.wait()
            except KeyboardInterrupt:
                proc.terminate()
                try:
                    return proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    return proc.wait()
        payload = supervisor.start(args.process_id)
    elif args.command == "stop":
        payload = supervisor.stop(args.process_id)
    elif args.command == "restart":
        payload = supervisor.restart(args.process_id)
    elif args.command == "logs":
        payload = supervisor.tail_logs(args.process_id, limit=max(1, min(int(args.limit), 2000)))
    elif args.command == "selftest":
        payload = supervisor.ftp_selftest(args.process_id)
    else:
        raise ValueError(f"Unsupported command: {args.command}")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
