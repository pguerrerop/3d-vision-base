from __future__ import annotations

from pathlib import Path

from vision_3d_acquisition.runtime.commanding import normalize_ftp_command
from vision_3d_acquisition.runtime.process_runner import TriSpectorFtpRuntime


def test_normalize_ftp_command_injects_write_and_root() -> None:
    out = normalize_ftp_command("python -m pyftpdlib -p 2121", Path("/tmp/uploads"), python_executable="/venv/bin/python")
    assert "-w" in out.command
    assert "-d" in out.command
    assert out.injected_write is True


def test_normalize_ftp_command_anonymous_mode() -> None:
    out = normalize_ftp_command("python -m pyftpdlib -w -p 2121", Path("/tmp/uploads"), auth_mode="anonymous")
    assert out.auth_mode == "anonymous"
    assert " -u " not in f" {out.command} "
    assert " -P " not in f" {out.command} "


def test_normalize_ftp_command_user_password_mode() -> None:
    out = normalize_ftp_command(
        "python -m pyftpdlib -w -p 2121",
        Path("/tmp/uploads"),
        auth_mode="user_password",
        username="trispector",
        password="secret",
    )
    assert out.auth_mode == "user_password"
    assert "-u" in out.command
    assert "-P" in out.command


def test_runtime_emits_upload_and_login_events_from_ftp_log_lines(tmp_path: Path, monkeypatch) -> None:
    runtime = TriSpectorFtpRuntime(
        process_id="trispector_ftp",
        source_id="trispector_ftp_0",
        ftp_command="python -m pyftpdlib -w -p 2121",
        upload_dir=tmp_path / "incoming",
        poll_interval_sec=0.1,
        stable_checks=1,
    )
    events: list[str] = []
    monkeypatch.setattr(runtime, "emit", lambda event_type, _message, **_kwargs: events.append(event_type))

    runtime._emit_ftp_connection_event("127.0.0.1 connected")
    runtime._emit_ftp_connection_event("username=trispector logged in")
    runtime._emit_ftp_connection_event("failed login username=bad")
    runtime._emit_ftp_connection_event("STOR sample.bin 120 bytes")

    assert "FTP_CLIENT_CONNECTED" in events
    assert "FTP_CLIENT_LOGIN_SUCCESS" in events
    assert "FTP_CLIENT_LOGIN_FAILED" in events
    assert "FTP_UPLOAD_STARTED" in events
