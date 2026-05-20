"""Manual GenTL/CTI validation utility.

This script intentionally lives outside pytest's collection tree because it
depends on local SDK files, camera hardware, and the host GenTL environment.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate a GenTL CTI producer and optionally acquire one buffer.",
    )
    parser.add_argument(
        "--cti",
        required=True,
        type=Path,
        help="Path to the GenTL CTI producer, for example vendor/sick/sdk_4_3/SICKGigEVisionTL.cti.",
    )
    parser.add_argument(
        "--device-index",
        type=int,
        default=0,
        help="Zero-based device index to use for acquisition. Default: 0.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=5.0,
        help="Timeout in seconds while waiting for one acquired buffer. Default: 5.0.",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="Only list devices detected through the CTI producer; do not acquire.",
    )
    return parser


def _format_device(index: int, info: object) -> str:
    return f"[{index}] {info}"


def _create_image_acquirer(harvester: object, device_index: int) -> object:
    try:
        return harvester.create(device_index)
    except TypeError:
        return harvester.create({"index": device_index})


def _stop_acquirer(image_acquirer: object) -> None:
    stop = getattr(image_acquirer, "stop", None)
    if callable(stop):
        stop()
    destroy = getattr(image_acquirer, "destroy", None)
    if callable(destroy):
        destroy()


def run_validation(args: argparse.Namespace) -> int:
    try:
        from harvesters.core import Harvester
    except ImportError as exc:
        print("ERROR: harvesters is not installed. Install project requirements first.")
        print(f"Details: {exc}")
        return 2

    cti_path = args.cti.expanduser()
    if not cti_path.exists():
        print(f"ERROR: CTI producer not found: {cti_path}")
        return 2
    if not cti_path.is_file():
        print(f"ERROR: CTI path is not a file: {cti_path}")
        return 2

    print("GenTL producers:")
    print(f"- {cti_path.resolve()}")

    harvester = Harvester()
    try:
        harvester.add_file(str(cti_path))
        harvester.update()

        device_info_list = list(harvester.device_info_list)
        print("Devices found:")
        if device_info_list:
            for index, info in enumerate(device_info_list):
                print(_format_device(index, info))
        else:
            print("- none")

        if args.list_devices:
            return 0

        if args.device_index < 0 or args.device_index >= len(device_info_list):
            print(
                "ERROR: device index "
                f"{args.device_index} is out of range for {len(device_info_list)} detected device(s)."
            )
            return 2

        image_acquirer = None
        try:
            print(f"Starting acquisition from device index {args.device_index}...")
            image_acquirer = _create_image_acquirer(harvester, args.device_index)
            image_acquirer.start()
            with image_acquirer.fetch(timeout=args.timeout) as buffer:
                payload = getattr(buffer, "payload", None)
                components = getattr(payload, "components", []) if payload is not None else []
                component_count = len(components) if components is not None else 0
                print("Acquisition succeeded.")
                print(f"Buffer payload components: {component_count}")
            return 0
        except Exception as exc:
            print("ERROR: acquisition failed.")
            print(f"Details: {exc}")
            return 1
        finally:
            if image_acquirer is not None:
                _stop_acquirer(image_acquirer)
    except Exception as exc:
        print("ERROR: CTI validation failed.")
        print(f"Details: {exc}")
        return 1
    finally:
        harvester.reset()


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return run_validation(args)


if __name__ == "__main__":
    raise SystemExit(main())
