from __future__ import annotations

from pathlib import Path

from vision_3d_acquisition.runtime.commanding import normalize_ftp_command, resolve_python_executable


def test_resolve_python_executable_prefers_current_or_fallback() -> None:
    resolution = resolve_python_executable()
    assert isinstance(resolution.executable, str)
    assert resolution.executable != ""


def test_normalize_python_module_command_and_inject_root(tmp_path: Path) -> None:
    upload = tmp_path / "incoming"
    out = normalize_ftp_command("python -m pyftpdlib -w -p 2121", upload, python_executable="/tmp/fake/python")
    assert out.normalized_python is True
    assert out.injected_root is True
    assert "-d" in out.command


def test_detect_ftp_root_mismatch(tmp_path: Path) -> None:
    upload = tmp_path / "incoming"
    wrong = tmp_path / "other"
    out = normalize_ftp_command(
        f"python3 -m pyftpdlib -w -d {wrong}",
        upload,
        python_executable="/tmp/fake/python",
    )
    assert out.root_mismatch is True
