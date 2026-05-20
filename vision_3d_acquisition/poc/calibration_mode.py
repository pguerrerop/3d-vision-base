from __future__ import annotations

import os
import sys
from pathlib import Path


AUTO_PLANE_WARNING = """WARNING:
No calibration file specified.
Using automatic plane estimation only.

Results may be:
- less stable
- non-repeatable
- sensitive to scene composition
"""


def calibration_mode(calibration: Path | str | None) -> str:
    return "calibrated" if calibration is not None else "auto"


def calibration_display(calibration: Path | str | None) -> str:
    return Path(calibration).name if calibration is not None else "NONE"


def calibration_status(calibration: Path | str | None) -> str:
    return "calibrated" if calibration is not None else "automatic plane estimation only"


def should_prompt_for_auto_mode(*, yes: bool, non_interactive: bool) -> bool:
    if yes or non_interactive:
        return False
    if os.environ.get("CI"):
        return False
    return sys.stdin.isatty() and sys.stdout.isatty()


def confirm_auto_mode(*, yes: bool, non_interactive: bool) -> bool:
    if not should_prompt_for_auto_mode(yes=yes, non_interactive=non_interactive):
        return True
    response = input("No calibration specified.\nContinue with automatic plane estimation? [y/N] ")
    return response.strip().lower() in {"y", "yes"}
