from __future__ import annotations

import shlex
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PythonResolution:
    executable: str
    used_fallback: bool
    warning: str | None = None


@dataclass(frozen=True)
class FtpCommandNormalization:
    command: str
    normalized_python: bool
    injected_root: bool
    root_mismatch: bool
    injected_write: bool
    auth_mode: str
    auth_applied: bool


def resolve_python_executable() -> PythonResolution:
    current = (sys.executable or "").strip()
    if current and Path(current).is_file():
        return PythonResolution(executable=current, used_fallback=False)
    for candidate in ("python3", "python"):
        resolved = shutil.which(candidate)
        if resolved:
            return PythonResolution(
                executable=resolved,
                used_fallback=True,
                warning=(
                    f"sys.executable is unavailable; falling back to '{candidate}' at '{resolved}'. "
                    "Activate your virtualenv to keep interpreter selection deterministic."
                ),
            )
    return PythonResolution(
        executable="python3",
        used_fallback=True,
        warning=(
            "Unable to resolve sys.executable or PATH python binaries; using 'python3'. "
            "Runtime startup may fail until a Python executable is available."
        ),
    )


def normalize_ftp_command(
    command: str,
    upload_dir: Path,
    *,
    python_executable: str | None = None,
    auth_mode: str = "anonymous",
    username: str | None = None,
    password: str | None = None,
) -> FtpCommandNormalization:
    tokens = shlex.split(command.strip())
    if not tokens:
        return FtpCommandNormalization(
            command=command,
            normalized_python=False,
            injected_root=False,
            root_mismatch=False,
            injected_write=False,
            auth_mode=auth_mode,
            auth_applied=False,
        )

    normalized_python = False
    resolved_python = python_executable or resolve_python_executable().executable
    if len(tokens) >= 3 and tokens[0] in {"python", "python3"} and tokens[1] == "-m":
        tokens[0] = resolved_python
        normalized_python = True

    injected_root = False
    root_mismatch = False
    injected_write = False
    auth_applied = False
    upload_str = str(upload_dir)
    if "-m" in tokens and "pyftpdlib" in tokens:
        if "-w" not in tokens:
            tokens.insert(len(tokens), "-w")
            injected_write = True
        if "-d" not in tokens:
            tokens.extend(["-d", upload_str])
            injected_root = True
        else:
            idx = tokens.index("-d")
            if idx + 1 < len(tokens):
                configured = str(Path(tokens[idx + 1]).expanduser())
                expected = str(Path(upload_str).expanduser())
                root_mismatch = configured != expected
        mode = str(auth_mode or "anonymous").strip().lower()
        if mode == "user_password":
            u = (username or "").strip()
            p = password or ""
            if u and p:
                if "-u" not in tokens:
                    tokens.extend(["-u", u])
                    auth_applied = True
                if "-P" not in tokens:
                    tokens.extend(["-P", p])
                    auth_applied = True
        else:
            mode = "anonymous"

    return FtpCommandNormalization(
        command=shlex.join(tokens),
        normalized_python=normalized_python,
        injected_root=injected_root,
        root_mismatch=root_mismatch,
        injected_write=injected_write,
        auth_mode=mode if "-m" in tokens and "pyftpdlib" in tokens else str(auth_mode or "anonymous"),
        auth_applied=auth_applied,
    )
