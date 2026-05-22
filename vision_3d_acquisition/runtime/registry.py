from __future__ import annotations

from pathlib import Path

from vision_3d_acquisition.api.settings import ApiSettings
from vision_3d_acquisition.runtime.commanding import normalize_ftp_command, resolve_python_executable
from vision_3d_acquisition.runtime.models import RuntimeProcessDefinition
from vision_3d_acquisition.runtime_config import RUNTIME_CONFIG_PATH, read_runtime_config


def build_runtime_registry(settings: ApiSettings) -> dict[str, RuntimeProcessDefinition]:
    runtime_config = read_runtime_config(RUNTIME_CONFIG_PATH).config.runtime.trispector_ftp
    upload_dir = Path(runtime_config.upload_dir)
    if not upload_dir.is_absolute():
        upload_dir = settings.data_dir.parent / upload_dir
    resolution = resolve_python_executable()
    python_exec = resolution.executable
    normalized_ftp = normalize_ftp_command(
        runtime_config.command,
        upload_dir=upload_dir,
        python_executable=python_exec,
        auth_mode=runtime_config.auth_mode,
        username=runtime_config.username,
        password=runtime_config.password,
    )
    command = [
        python_exec,
        "-m",
        "vision_3d_acquisition.runtime.process_runner",
        "trispector_ftp_runtime",
        "--source-id",
        runtime_config.source_id,
        "--ftp-command",
        normalized_ftp.command,
        "--upload-dir",
        str(upload_dir),
        "--poll-interval-sec",
        str(runtime_config.poll_interval_sec),
        "--stable-checks",
        str(runtime_config.stable_checks),
        "--auth-mode",
        runtime_config.auth_mode,
    ]
    if runtime_config.username:
        command.extend(["--username", runtime_config.username])
    if runtime_config.password:
        command.extend(["--password", runtime_config.password])
    return {
        "trispector_ftp": RuntimeProcessDefinition(
            process_id="trispector_ftp",
            process_type="trispector_ftp_runtime",
            command=command,
            source_id=runtime_config.source_id,
            enabled=runtime_config.enabled,
            config={
                "enabled": runtime_config.enabled,
                "command": runtime_config.command,
                "normalized_command": normalized_ftp.command,
                "upload_dir": str(upload_dir),
                "poll_interval_sec": runtime_config.poll_interval_sec,
                "stable_checks": runtime_config.stable_checks,
                "ftp_root_injected": normalized_ftp.injected_root,
                "ftp_root_mismatch": normalized_ftp.root_mismatch,
                "ftp_command_normalized": normalized_ftp.normalized_python,
                "ftp_write_enabled": normalized_ftp.injected_write or (" -w" in f" {normalized_ftp.command} "),
                "auth_mode": normalized_ftp.auth_mode,
                "username": runtime_config.username,
                "password": runtime_config.password,
                "auth_applied": normalized_ftp.auth_applied,
                "python_resolution_warning": resolution.warning,
            },
        )
    }
